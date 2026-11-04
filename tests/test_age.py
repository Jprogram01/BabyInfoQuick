from babyinfoquick.age import AgeBand, bands_for_months, extract_age_bands


def test_point_age_in_weeks_maps_to_newborn():
    assert extract_age_bands("my 3 week old won't sleep") == [AgeBand.NEWBORN]


def test_point_age_in_months():
    assert extract_age_bands("solid foods for a 6 month old") == [AgeBand.INFANT_4_6]


def test_range_spans_bands():
    assert extract_age_bands("feeding schedule 4-6 months") == [AgeBand.INFANT_4_6]


def test_years_map_to_toddler():
    assert extract_age_bands("what about a 2 year old") == [AgeBand.TODDLER_24_36]


def test_keyword_fallback_newborn():
    assert extract_age_bands("newborn cord care") == [AgeBand.NEWBORN]


def test_keyword_fallback_toddler():
    assert extract_age_bands("toddler tantrums") == [
        AgeBand.TODDLER_12_24,
        AgeBand.TODDLER_24_36,
    ]


def test_no_age_signal():
    assert extract_age_bands("how do I sterilize bottles") == []


def test_bands_for_months_is_half_open():
    assert bands_for_months(0, 1) == [AgeBand.NEWBORN]
    assert AgeBand.INFANT_1_3 in bands_for_months(1, 4)
