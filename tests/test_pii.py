from src.utils.pii import mask_pii


def test_masks_emails_and_phones():
    out = mask_pii("Mail jane.doe@gmail.com or call +1 (415) 555-2671 / 020 7946 0958")
    assert "jane.doe" not in out and "555-2671" not in out and "7946" not in out
    assert out.count("[EMAIL]") == 1 and out.count("[PHONE]") == 2


def test_keeps_product_support_email_and_ordinary_numbers():
    text = "Write help@aurorasync.example. You get 2 GB, 30 days, $4.99, 500 members, 2024-05-12."
    assert mask_pii(text) == text


def test_handles_empty():
    assert mask_pii("") == "" and mask_pii(None) is None
