BEGIN;

CREATE TABLE IF NOT EXISTS compounding (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

INSERT OR IGNORE INTO compounding (id, type, params_json)
VALUES (0, 'NONE', '{}');

COMMIT;
