-- 02_quality_checks.sql — data quality audit queries.
-- Each query is named with a "-- name: <key>" tag and run by
-- src/data/quality_audit.py, which assembles reports/data_quality_report.md.
-- "?" is the dataset's missing marker.

-- name: row_count
SELECT count(*) AS n_encounters FROM encounters_raw;

-- name: unique_patients
SELECT count(DISTINCT patient_nbr) AS n_patients FROM encounters_raw;

-- name: duplicate_encounter_ids
SELECT count(*) AS n_duplicate_encounter_ids
FROM (
    SELECT encounter_id FROM encounters_raw GROUP BY encounter_id HAVING count(*) > 1
);

-- name: encounters_per_patient
WITH per_patient AS (
    SELECT patient_nbr, count(*) AS n FROM encounters_raw GROUP BY patient_nbr
)
SELECT
    min(n)      AS min_encounters,
    max(n)      AS max_encounters,
    round(avg(n), 3) AS mean_encounters,
    count(*)    AS n_patients,
    count(*) FILTER (WHERE n = 1) AS patients_with_1,
    count(*) FILTER (WHERE n = 2) AS patients_with_2,
    count(*) FILTER (WHERE n >= 3) AS patients_with_3plus
FROM per_patient;

-- name: readmitted_distribution
SELECT
    readmitted,
    count(*) AS n,
    round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
FROM encounters_raw
GROUP BY readmitted
ORDER BY n DESC;

-- name: gender_distribution
SELECT gender, count(*) AS n
FROM encounters_raw
GROUP BY gender
ORDER BY n DESC;

-- name: a1cresult_distribution
SELECT A1Cresult AS value, count(*) AS n
FROM encounters_raw
GROUP BY A1Cresult
ORDER BY n DESC;

-- name: max_glu_serum_distribution
SELECT max_glu_serum AS value, count(*) AS n
FROM encounters_raw
GROUP BY max_glu_serum
ORDER BY n DESC;

-- name: death_hospice_discharges
-- Encounters whose discharge disposition means death/hospice (to be excluded).
SELECT
    CAST(e.discharge_disposition_id AS INTEGER) AS discharge_disposition_id,
    m.description,
    count(*) AS n
FROM encounters_raw e
LEFT JOIN ids_mapping m
    ON m.mapping_type = 'discharge_disposition_id'
   AND m.id = CAST(e.discharge_disposition_id AS INTEGER)
WHERE CAST(e.discharge_disposition_id AS INTEGER) IN (11, 13, 14, 19, 20, 21)
GROUP BY 1, 2
ORDER BY 1;
