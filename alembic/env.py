import sys

# alembic/env.py (Snippet of the modified sections)
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

import os
from dotenv import load_dotenv
from sqlmodel import SQLModel



# 1. IMPORT YOUR MODELS HERE. This attaches your tables to SQLModel.metadata
import database.models

# 2. Load the .env file
load_dotenv()

# this is the Alembic Config object
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 3. OVERRIDE THE URL. Force Alembic to use the .env URL instead of alembic.ini
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL must be set in your .env file for migrations.")
config.set_main_option("sqlalchemy.url", database_url)

# 4. POINT TO SQLMODEL METADATA
target_metadata = SQLModel.metadata

# ... keep the rest of the file (run_migrations_offline, run_migrations_online) as is.