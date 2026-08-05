-- Runs once, on first boot of an empty Postgres volume.
-- Keeps pytest off the development database.
CREATE DATABASE sil_agent_test OWNER sil;
