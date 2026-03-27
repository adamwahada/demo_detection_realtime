# Audit Couche Données — `demo_detection_realtime`

---

## 1. BASE DE DONNÉES

### 1.1 Moteur utilisé et configuration de connexion

Le projet tente **d'abord PostgreSQL**, et se rabat sur **SQLite** si la connexion échoue ou si `psycopg2` n'est pas installé.

La logique de sélection est dans `db_writer.py` → `DBWriter._detect_backend()` :

```python
try:
    import psycopg2           # si l'import échoue → SQLite direct
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, ...)
    conn.close()
    return "postgres"         # succès → backend PostgreSQL
except Exception:
    return "sqlite"           # échec → fallback SQLite
```

**Configuration** (fichier `db_config.py`) — toutes les valeurs sont **codées en dur** :

| Paramètre | Valeur | Notes |
|---|---|---|
| `DB_HOST` | `"localhost"` | Pas de support de variable d'environnement |
| `DB_PORT` | `5432` | Port PostgreSQL standard |
| `DB_NAME` | `"tracking"` | Nom de la base |
| `DB_USER` | `"tracking_user"` | |
| `DB_PASSWORD` | `"changeme"` | Mot de passe placeholder, en clair dans le code |
| `STATS_ENABLED_DEFAULT` | `False` | L'enregistrement est désactivé au démarrage |
| `SNAPSHOT_EVERY_N_PACKETS` | `25` | Fréquence d'écriture des snapshots |
| `WRITE_QUEUE_MAXSIZE` | `10000` | Taille maximale de la file d'événements |

**Chemin SQLite** (calculé dynamiquement dans `db_writer.py`) :
```
<dossier du script>/data/tracking_demo.db
```

Fichiers présents sur le disque au moment de l'audit :
```
demo_detection_realtime/data/
    tracking_demo.db
    tracking_demo.db-shm    ← WAL shared memory
    tracking_demo.db-wal    ← WAL write-ahead log (mode WAL actif)
```

---

### 1.2 Tables, colonnes, types et contraintes

Le schéma est défini deux fois en miroir :
- `db_schema.sql` — fichier séparé (référence documentaire uniquement)
- `_SQLITE_SCHEMA` — chaîne Python dans `db_writer.py` (seule version réellement exécutée)

#### Table `sessions`

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,      -- UUID v4 généré à l'ouverture
    started_at      TEXT,                  -- horodatage ISO 8601 (chaîne, pas DATETIME)
    ended_at        TEXT,                  -- NULL jusqu'à la fermeture de session
    checkpoint_id   TEXT DEFAULT '',       -- ex: "anomaly", "barcode_date"
    camera_source   TEXT DEFAULT '',       -- ex: chemin fichier ou URL RTSP
    total           INTEGER DEFAULT 0,     -- total paquets comptés
    ok_count        INTEGER DEFAULT 0,
    nok_no_barcode  INTEGER DEFAULT 0,
    nok_no_date     INTEGER DEFAULT 0
);
```

**Contraintes :** `PRIMARY KEY` uniquement. Pas de `NOT NULL` sur les colonnes métier, pas de `CHECK`, pas de `UNIQUE` secondaire.

#### Table `stats_snapshots`

```sql
CREATE TABLE IF NOT EXISTS stats_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,                  -- référence sessions.id (pas de FK déclarée)
    captured_at     TEXT,                  -- horodatage ISO 8601
    total           INTEGER DEFAULT 0,
    ok_count        INTEGER DEFAULT 0,
    nok_no_barcode  INTEGER DEFAULT 0,
    nok_no_date     INTEGER DEFAULT 0,
    nok_rate        REAL DEFAULT 0.0       -- calculé à l'écriture : (total-ok)/total*100
);

CREATE INDEX IF NOT EXISTS idx_snapshots_session ON stats_snapshots (session_id);
```

**Contraintes :** `PRIMARY KEY AUTOINCREMENT` + index sur `session_id`. Pas de `FOREIGN KEY` déclarée vers `sessions(id)`.

---

### 1.3 Initialisation de la base

L'initialisation est déclenchée dans `DBWriter.__init__()` si le backend détecté est SQLite :

```python
def _init_sqlite(self):
    os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)   # crée data/ si absent
    conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")                      # mode WAL activé
    conn.executescript(_SQLITE_SCHEMA)                           # CREATE TABLE IF NOT EXISTS
    conn.commit()
    self._sqlite_conn = conn
```

**Pas de système de migration.** Si le schéma change, les tables existantes ne sont pas modifiées automatiquement — seul `CREATE TABLE IF NOT EXISTS` est exécuté, ce qui n'applique pas d'`ALTER TABLE`.

Pour PostgreSQL, **il n'y a aucun code d'initialisation** — les tables doivent exister préalablement. Si elles sont absentes, toutes les requêtes échoueront silencieusement (les erreurs sont catchées et l'objet se dégrade sur `None`).

---

### 1.4 Toutes les requêtes SQL, regroupées par table

#### Table `sessions`

| Opération | Méthode | SQL |
|---|---|---|
| INSERT | `open_session()` | `INSERT INTO sessions (id, started_at, checkpoint_id, camera_source) VALUES (?, ?, ?, ?)` |
| UPDATE (clôture) | `close_session()` | `UPDATE sessions SET ended_at=?, total=?, ok_count=?, nok_no_barcode=?, nok_no_date=? WHERE id=?` |
| SELECT par id | `get_session_kpis()` | `SELECT * FROM sessions WHERE id = ?` |
| SELECT liste | `list_sessions()` | `SELECT id, started_at, ended_at, checkpoint_id, camera_source, total, ok_count, nok_no_barcode, nok_no_date FROM sessions ORDER BY started_at DESC LIMIT ?` |

#### Table `stats_snapshots`

| Opération | Méthode | SQL |
|---|---|---|
| INSERT | `_write_snapshot()` | `INSERT INTO stats_snapshots (session_id, captured_at, total, ok_count, nok_no_barcode, nok_no_date, nok_rate) VALUES (?,?,?,?,?,?,?)` |
| SELECT par session | `list_snapshots()` | `SELECT id, session_id, captured_at, total, ok_count, nok_no_barcode, nok_no_date, nok_rate FROM stats_snapshots WHERE session_id = ? ORDER BY captured_at ASC LIMIT ?` |

**Paramètres PostgreSQL :** les mêmes requêtes utilisent `%s` à la place de `?`.

---

### 1.5 Le DBWriter : fonctionnement, write_queue, événements

#### Architecture générale

```
Detector thread                DBWriter thread
──────────────                 ──────────────
tracking_state.py              db_writer.py
                               
write_queue.put_nowait({...})  →  _run() loop
                                      │
                                      ├── type="snapshot" → _write_snapshot()
                                      └── type="stop"     → break
```

- Le `DBWriter` est instancié dans `TrackingState.__init__()` (ligne 156).
- Le thread `DBWriter` est **démarré en lazy** : seulement au premier appel `set_stats_recording(True)`, via `_db_writer_started` flag.
- Thread nommé `"DBWriter"`, daemon=True.
- La queue est `queue.Queue(maxsize=10000)`.

#### Appels **directs** (hors queue, bloquants sur le thread appelant)

| Méthode | Appelée depuis | Nature |
|---|---|---|
| `open_session()` | `set_stats_recording(True)` → Flask thread | Écrit synchronement en SQLite ou PG |
| `close_session()` | `set_stats_recording(False)` → Flask thread | Écrit synchronement en SQLite ou PG |
| `get_session_kpis()` | `api_stats_session()` → Flask thread | Lecture synchrone |
| `list_sessions()` | `api_stats_sessions()` → Flask thread | Lecture synchrone |
| `list_snapshots()` | `api_stats_session_snapshots()` → Flask thread | Lecture synchrone |

**Toutes les lectures ET les ouvertures/fermetures de session passent directement par la connexion SQLite**, protégée par `_sqlite_lock` (threading.Lock). La queue ne sert que pour les snapshots.

#### Événements dans la queue

Il n'existe qu'**un seul type d'événement** :

```python
{
    "type": "snapshot",
    "session_id": str,        # UUID de la session active
    "total": int,             # total_packets à l'instant du snapshot
    "ok_count": int,          # output_fifo.count("OK") — O(N)
    "nok_no_barcode": int,    # compteur en mémoire
    "nok_no_date": int,       # compteur en mémoire
}
```

La gestion d'erreur dans `_run()` reset `_pg_conn = None` en cas d'erreur PG, mais **n'enqueue pas l'événement en échec** — les snapshots perdus ne sont pas réessayés.

Si `is_active` est False au moment où le worker dépile l'événement, l'événement est **silencieusement jeté** (`task_done()` sans écriture).

---

### 1.6 Snapshots : déclenchement et contenu

#### Quand un snapshot est-il pris ?

Un snapshot est enqueué dans **deux endroits** du code, tous deux dans `tracking_state.py` :

| Contexte | Condition | Fichier, ligne approx. |
|---|---|---|
| Mode **tracking** (exit line crossing) | `self._stats_active AND total_packets % 25 == 0` | `tracking_state.py` ligne ~1597 |
| Mode **anomaly** (decision zone) | `self._stats_active AND total_packets % 25 == 0` | `tracking_state.py` ligne ~1340 |

#### Contenu sauvegardé par snapshot

| Champ | Source | Mode tracking | Mode anomaly |
|---|---|---|---|
| `session_id` | `_db_session_id` | ✅ | ✅ |
| `total` | `self.total_packets` | ✅ | ✅ |
| `ok_count` | `output_fifo.count("OK")` | ✅ | ✅ |
| `nok_no_barcode` | `self._nok_no_barcode` | ✅ compteur réel | ⚠️ hardcodé à `0` |
| `nok_no_date` | `self._nok_no_date` | ✅ compteur réel | ⚠️ hardcodé à `0` |

Le champ `nok_rate` est **calculé au moment de l'écriture** dans `_write_snapshot()` : `(total - ok_count) / total * 100`.

#### Quand la session est-elle fermée ?

`close_session()` est appelé uniquement via `set_stats_recording(False)` (action manuelle depuis l'UI). **Il n'y a pas de fermeture automatique à l'arrêt du serveur** — si le processus est tué, `ended_at` reste `NULL` dans la base.

---

## 2. STOCKAGE IMAGES / CAPTURES

### 2.1 Chemin et structure des dossiers

Les fichiers ne sont sauvegardés que pour les paquets **NOK en mode anomaly**. La structure est :

```
demo_detection_realtime/
└── anomalie/
    ├── 3/                  ← paquet NOK #3
    │   ├── scan_1.png
    │   ├── scan_2.png
    │   ├── scan_3.png      ← jusqu'à ad_max_scans crops (défaut = 5)
    │   └── scans.csv
    ├── 52/
    │   ├── scan_1.png
    │   └── scans.csv
    └── 54/ 55/ 56/ 57/ 58/ ...
```

Dossiers existants à l'audit : `3, 52, 54, 55, 56, 57, 58`

Le chemin de base est calculé dynamiquement :
```python
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anomalie")
pkt_dir = os.path.join(base, str(pkt_num))
```

---

### 2.2 Fonction d'écriture et thread

| Élément | Valeur |
|---|---|
| Fonction principale | `_save_nok_packet(pkt_num, tstate, checkpoint)` dans `tracking_state.py` |
| Thread appelant | Thread daemon nommé `AD-Save-{pkt_num}` — fire-and-forget |
| Lancé depuis | `_save_nok_packet_bg()` appelé dans le YOLODetector thread |
| Concurrence | Chaque paquet NOK lance un thread indépendant — pas de limite ni de pool |

Le `YOLODetector` fait une copie snapshot des données avant de lancer le thread :
```python
data = {
    'results': list(tstate.get('results', [])),  # copie
    'scores':  list(tstate.get('scores', [])),
    'crops':   list(tstate.get('crops', [])),     # refs vers numpy arrays
}
```

---

### 2.3 Ce qui est sauvegardé par paquet NOK

#### Images (`.png`)

- Un fichier `scan_<idx>.png` par crop collecté pendant la zone de scan
- Le crop est la version **letterboxée 640×640 RGB** reconvertie en BGR pour `cv2.imwrite`
- Format PNG (sans compression lossy)
- Nombre de fichiers = `len(tstate['crops'])` ≤ `ad_max_scans` (défaut 5)

#### CSV (`scans.csv`)

```
scan,score,threshold,is_defective
1,6234.50,5000.00,YES
2,5891.32,5000.00,YES
3,4211.10,5000.00,NO

DECISION,NOK,defective_scans=2/3,strategy=MAJORITY
```

Colonnes : `scan`, `score`, `threshold`, `is_defective`  
Ligne de conclusion : `DECISION`, verdict final, ratio scans défectueux / total, stratégie utilisée.

---

### 2.4 Référencement en base de données

**Le chemin du dossier `anomalie/<pkt_num>/` n'est jamais enregistré en base de données.**

Il n'existe aucun lien entre les enregistrements `sessions` / `stats_snapshots` et les fichiers sur disque. La réconciliation doit se faire manuellement par numéro de paquet.

---

## 3. POSTGRESQL — ÉTAT DE L'IMPLÉMENTATION

### 3.1 Ce qui est implémenté

L'intégration PostgreSQL est **entièrement codée** mais utilise une connexion persistante avec reconnexion manuelle (pas de pool). Voici ce qui existe :

| Élément | Statut |
|---|---|
| Import conditionnel `psycopg2` | ✅ présent avec flag `_PSYCOPG2_AVAILABLE` |
| Détection automatique du backend au démarrage | ✅ timeout 2 s |
| Toutes les requêtes dupliquées `?` / `%s` | ✅ dans toutes les méthodes |
| `_get_pg_conn()` avec retry (3 tentatives, sleep 1 s) | ✅ |
| Reconnexion automatique si connexion perdue | ✅ via `_pg_conn = None` reset |
| `cursor_factory=RealDictCursor` pour les lectures | ✅ |

### 3.2 Limitations et absences

| Problème | Détail |
|---|---|
| Mot de passe en clair dans le code | `DB_PASSWORD = "changeme"` dans `db_config.py` — pas de lecture depuis variable d'environnement (`os.environ`) |
| Pas de `DATABASE_URL` | Aucun support de l'URL de connexion unifiée (`postgresql://user:pass@host/db`) |
| Pas d'initialisation du schéma pour PG | Si les tables n'existent pas, toutes les opérations échouent silencieusement. `db_schema.sql` existe mais n'est pas exécuté automatiquement. |
| Pas de pool de connexions | Une seule connexion persistante partagée entre lectures (Flask threads) et écritures (DBWriter thread), protégée uniquement par les blocs `try/except` — pas de `threading.Lock` sur `_pg_conn` |
| `psycopg2` absent des requirements | `requirements_web.txt` ne liste pas `psycopg2` (à vérifier — si absent, le fallback SQLite s'active silencieusement) |

### 3.3 Code commenté / flags liés à la DB

- Aucun code commenté directement lié à la DB dans les fichiers audités.
- Le flag `STATS_ENABLED_DEFAULT = False` dans `db_config.py` désactive silencieusement tout l'enregistrement au démarrage — aucun avertissement n'est affiché si l'utilisateur oublie d'activer manuellement.

---

## 4. DOCKER

### 4.1 État du Docker dans `demo_detection_realtime`

**Il n'y a pas de `docker-compose.yml` dans `demo_detection_realtime`.** Seul `tracking_live2/` en possède un.

```
demo_detection_realtime/
    requirements_web.txt     ← liste des dépendances Python
    (aucun Dockerfile)
    (aucun docker-compose.yml)
```

Le projet tourne donc directement avec `python web_server_backend_v2.py` dans l'environnement pyenv `tracking_live-env`.

### 4.2 Impact sur la couche données

Sans Docker, les implications sur la couche données sont :

| Point | État actuel |
|---|---|
| PostgreSQL | Doit être installé et démarré manuellement sur le host |
| SQLite | Fichier créé dans `demo_detection_realtime/data/` — persistant tant que le dossier existe |
| Dossier `anomalie/` | Créé à la volée à la première sauvegarde — chemin absolu ancré dans `__file__` |
| Variables d'environnement | Inexistantes — tout est hardcodé dans `db_config.py` |

---

## 5. GAPS ET INCOHÉRENCES

### 5.1 Données calculées à la volée non persistées

| Donnée | Où elle est calculée | Persistée ? |
|---|---|---|
| `output_fifo` (liste OK/NOK) | En mémoire dans `TrackingState` | ❌ Perdu au redémarrage |
| `packet_numbers` (dict tid→num) | En mémoire dans `TrackingState` | ❌ Perdu au redémarrage |
| `total_packets` | Compteur en mémoire | ❌ Remis à 0 à chaque session |
| `packets_crossed_line` (set) | En mémoire | ❌ |
| `nok_no_barcode` / `nok_no_date` | Compteurs en mémoire | ⚠️ Sauvegardés **uniquement** dans le snapshot final via `close_session()` |
| `ok_count` calculé dans snapshot | `output_fifo.count("OK")` — O(N) | ⚠️ Recalculé à chaque snapshot, pas incrémenté |

### 5.2 Colonnes définies mais renseignées à zéro en mode anomaly

En mode anomaly, le snapshot envoyé à la queue contient :
```python
"nok_no_barcode": 0,   # hardcodé
"nok_no_date": 0,      # hardcodé
```

Les colonnes `nok_no_barcode` et `nok_no_date` de la table `stats_snapshots` sont donc **toujours à 0 pour les sessions anomaly**, même si des paquets défectueux ont été comptés.

### 5.3 Pas de FOREIGN KEY en SQLite

```sql
-- La contrainte suivante n'est PAS déclarée :
-- FOREIGN KEY (session_id) REFERENCES sessions(id)
```

Des snapshots orphelins peuvent s'accumuler si une session est supprimée manuellement.

### 5.4 Dates stockées en TEXT

`started_at`, `ended_at`, `captured_at` sont des `TEXT` au format `'%Y-%m-%dT%H:%M:%S'` (pas de type `DATETIME` ou `TIMESTAMP WITH TIME ZONE`). Pas de timezone, pas de tri natif SQL par type date (fonctionne car le format ISO est tri-compatible).

### 5.5 Session non fermée à l'arrêt du serveur

Si le processus est tué (SIGKILL, SIGTERM, crash), `ended_at` reste `NULL` et `total/ok_count/nok_*` restent à leurs valeurs de la dernière écriture (pas forcément à jour). `close_session()` n'est pas appelé dans un handler `atexit` ou signal.

### 5.6 Lien manquant entre DB et fichiers disque

Pour un paquet NOK en mode anomaly :
- Le fichier est dans `anomalie/<pkt_num>/`
- La DB contient uniquement des compteurs agrégés dans `stats_snapshots`
- **Il n'existe aucune table `nok_packets`** enregistrant : `pkt_num`, `session_id`, `is_defective`, `score`, `file_path`

Le `pkt_num` n'est pas enregistré en DB — la réconciliation entre une session DB et les dossiers sur disque est impossible sans inférence manuelle.

### 5.7 `export_csv`

Il n'existe **pas de fonction `export_csv`** dans `db_writer.py` de `demo_detection_realtime` (contrairement à `tracking_live2`). L'API Flask n'expose pas d'endpoint d'export CSV des sessions.
