import pytest

from app.generator import GenerationError
from app.quota import DailyQuota

from .conftest import NO_FACE, FakeGenerator, solid

DAY = 24 * 3600


def post(client, upload, **kwargs):
    return client.post("/api/makeup", **upload(**kwargs))


# DailyQuota on its own


def test_reserve_refund_and_rolling_window():
    now = [0.0]
    quota = DailyQuota(3, clock=lambda: now[0])
    assert quota.remaining("1.2.3.4") == 3
    assert quota.reserve("1.2.3.4", 2) == 2
    assert quota.reserve("1.2.3.4", 3) == 1  # only one left
    assert quota.reserve("1.2.3.4", 1) == 0
    quota.refund("1.2.3.4", 1)
    assert quota.remaining("1.2.3.4") == 1
    assert quota.remaining("5.6.7.8") == 3  # other visitors are separate
    now[0] = DAY - 1
    assert quota.retry_after("1.2.3.4") == 0  # one call still free
    quota.reserve("1.2.3.4", 1)
    assert quota.retry_after("1.2.3.4") == 1
    now[0] = DAY + 1
    assert quota.remaining("1.2.3.4") == 2  # the two calls at time 0 expired; the one at DAY - 1 remains


def test_exempt_ip_has_no_limit():
    quota = DailyQuota(3, exempt=["9.9.9.9"])
    assert quota.remaining("9.9.9.9") is None
    assert quota.reserve("9.9.9.9", 100) is None


def test_ips_are_not_stored_in_plain_text():
    quota = DailyQuota(3)
    quota.reserve("203.0.113.7", 1)
    assert all("203.0.113.7" not in key for key in quota._calls)


# The API with a limit of 3 calls a day


def test_fourth_try_is_refused_without_calling_the_model(client_for, upload):
    generator = FakeGenerator(0.9, 0.9, 0.9, 0.9)
    client = client_for(generator, daily_generation_limit=3)
    left = [post(client, upload).json()["tries_left"] for _ in range(3)]
    assert left == [2, 1, 0]
    response = post(client, upload)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "daily_limit"
    assert int(response.headers["retry-after"]) > 0
    assert generator.calls == 3


def test_every_model_call_counts_including_retries(client_for, upload):
    generator = FakeGenerator(0.1, 0.1, 0.1, 0.9)
    client = client_for(generator, daily_generation_limit=3, max_attempts=3)
    assert post(client, upload).json()["attempts"] == 3
    assert post(client, upload).json()["error"]["code"] == "daily_limit"


def test_retries_are_capped_by_the_tries_left(client_for, upload):
    generator = FakeGenerator(0.9, 0.1, 0.1, 0.1)
    client = client_for(generator, daily_generation_limit=3, max_attempts=3)
    post(client, upload)  # passes first time: 1 used
    body = post(client, upload).json()  # fails every time but only 2 calls are left
    assert body["attempts"] == 2 and body["tries_left"] == 0
    assert generator.calls == 3


def test_rejected_uploads_do_not_use_a_try(client_for, upload):
    generator = FakeGenerator(0.9, 0.9, 0.9)
    client = client_for(generator, daily_generation_limit=3)
    for _ in range(5):
        assert post(client, upload, data=solid(NO_FACE, fmt="JPEG")).status_code == 422
    assert [post(client, upload).status_code for _ in range(3)] == [200, 200, 200]


def test_provider_quota_errors_are_refunded(client_for, upload):
    generator = FakeGenerator(GenerationError("OpenAI HTTP 429", retryable=False, quota=True), 0.9)
    client = client_for(generator, daily_generation_limit=1)
    assert post(client, upload).status_code == 503
    assert post(client, upload).status_code == 200


@pytest.mark.parametrize("requests", [6])
def test_exempt_ip_is_never_limited(client_for, upload, requests):
    generator = FakeGenerator(*[0.9] * requests)
    client = client_for(generator, daily_generation_limit=3, quota_exempt_ips=["testclient"])
    bodies = [post(client, upload).json() for _ in range(requests)]
    assert all(b["tries_left"] is None for b in bodies)
    assert generator.calls == requests


def test_limit_zero_turns_the_quota_off(client_for, upload):
    client = client_for(FakeGenerator(0.9, 0.9, 0.9, 0.9), daily_generation_limit=0)
    assert all(post(client, upload).status_code == 200 for _ in range(4))
