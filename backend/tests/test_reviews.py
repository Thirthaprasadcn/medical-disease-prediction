from backend.db.reviews import export_feedback, get_report_reviews, set_signoff, upsert_finding_review


def test_review_roundtrip(tmp_path, monkeypatch):
    db = tmp_path / "reviews.db"
    monkeypatch.setattr("backend.db.reviews.DB_PATH", db)
    saved = upsert_finding_review("scan1", 1, "accept", reviewer="doc")
    assert saved["decision"] == "accept"
    set_signoff("scan1", "signed", reviewer="doc")
    data = get_report_reviews("scan1")
    assert data["findings"][0]["decision"] == "accept"
    assert data["signoff"]["status"] == "signed"
    assert export_feedback()
