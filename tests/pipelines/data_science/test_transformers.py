import pandas as pd

from ieee_cis_fraud_detection.pipelines.data_science.transformers import (
    RareCategoryGrouper,
)


def test_rare_category_grouper_groups_rare_unseen_and_missing_values():
    train = pd.DataFrame(
        {
            "DeviceInfo": [
                "ios",
                "ios",
                "ios",
                "android",
                "android",
                "linux",
                None,
            ],
            "P_emaildomain": [
                "gmail.com",
                "gmail.com",
                "gmail.com",
                "yahoo.com",
                "proton.me",
                "proton.me",
                None,
            ],
        }
    )
    test = pd.DataFrame(
        {
            "DeviceInfo": ["ios", "linux", "windows", None],
            "P_emaildomain": ["gmail.com", "yahoo.com", "outlook.com", None],
        }
    )

    grouper = RareCategoryGrouper(min_frequency=0.3)
    transformed = grouper.fit(train).transform(test)

    assert transformed.to_dict(orient="list") == {
        "DeviceInfo": ["ios", "Rare", "Rare", "Rare"],
        "P_emaildomain": ["gmail.com", "Rare", "Rare", "Rare"],
    }


def test_rare_category_grouper_does_not_mutate_input_data():
    data = pd.DataFrame({"card4": ["visa", "visa", "mastercard", None]})
    original = data.copy(deep=True)

    grouper = RareCategoryGrouper(min_frequency=0.5)
    grouper.fit_transform(data)

    pd.testing.assert_frame_equal(data, original)
