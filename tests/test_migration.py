from sqlalchemy import text

EXPECTED_TABLES = {
    "policy_document",
    "coverage",
    "peril",
    "coverage_peril",
    "exclusion",
}


async def test_migration_created_all_tables(db_session):
    """`alembic upgrade head` (run in the container fixture) created the schema."""
    result = await db_session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
    )
    tables = {row[0] for row in result}
    assert EXPECTED_TABLES <= tables
