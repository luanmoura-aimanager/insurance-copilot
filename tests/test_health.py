async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_health_db(client):
    r = await client.get("/health/db")
    assert r.status_code == 200
    assert r.json() == {"db": "ok"}
