def test_create_tables_is_idempotent(tmp_weather_db):
    tmp_weather_db.createTables()
    tmp_weather_db.createTables()


def test_historical_daily_data_round_trip(tmp_weather_db):
    tmp_weather_db.insert_weather_stations([
        ("TEST", "ST1", "Test Station", "Test Region", 40.0, -3.0, 600),
    ])
    tmp_weather_db.insert_historical_daily_data([
        ("2021-01-01", "TEST", "ST1", 5.0, 15.0, 10.0, "06:00", "16:00",
         0.0, 10.0, 20.0, "N", "14:00", 40, 70, 55, "15:00", "05:00",
         1010.0, 1020.0, "13:00", "03:00"),
        ("2021-01-02", "TEST", "ST1", 4.0, 14.0, 9.0, "06:00", "16:00",
         2.5, 12.0, 22.0, "NE", "14:00", 45, 75, 60, "15:00", "05:00",
         1008.0, 1018.0, "13:00", "03:00"),
    ])

    result = tmp_weather_db.get_historical_daily_data("TEST", "ST1", "2021-01-01", "2021-01-02")

    assert len(result) == 2
    assert result[0]["date"] == "2021-01-01"
    assert result[0]["temperature_mean"] == 10.0
    assert result[1]["precipitation"] == 2.5


def test_historical_daily_data_filters_by_date_range(tmp_weather_db):
    tmp_weather_db.insert_weather_stations([("TEST", "ST1", "Test Station", None, 40.0, -3.0, 600)])
    tmp_weather_db.insert_historical_daily_data([
        (f"2021-01-{day:02d}", "TEST", "ST1") + (None,) * 19
        for day in range(1, 6)
    ])

    result = tmp_weather_db.get_historical_daily_data("TEST", "ST1", "2021-01-02", "2021-01-03")

    assert [r["date"] for r in result] == ["2021-01-02", "2021-01-03"]


def test_generation_job_round_trip(tmp_weather_db):
    job_ids = tmp_weather_db.insert_generation_jobs([
        (40.0, -3.0, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ])
    assert len(job_ids) == 1
    job_id = job_ids[0]

    info = tmp_weather_db.get_generation_job_info(job_id)
    assert info["latitude"] == 40.0
    assert info["historicalStartDate"] == "2018-01-01"


def test_generation_job_info_returns_none_for_unknown_id(tmp_weather_db):
    assert tmp_weather_db.get_generation_job_info(9999) is None


def test_monthly_predictions_round_trip(tmp_weather_db):
    job_id = tmp_weather_db.insert_generation_jobs([
        (40.0, -3.0, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ])[0]

    tmp_weather_db.insert_monthly_predictions([
        (job_id, 2021, 1, "temperature_mean", 5.0, 10.0, 15.0),
        (job_id, 2021, 1, "precipitation", 0.0, 2.0, 5.0),
    ])

    predictions = tmp_weather_db.get_monthly_predictions(job_id)

    assert len(predictions) == 2
    variables = {p["variable"] for p in predictions}
    assert variables == {"temperature_mean", "precipitation"}


def test_generated_daily_and_hourly_round_trip(tmp_weather_db):
    job_id = tmp_weather_db.insert_generation_jobs([
        (40.0, -3.0, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ])[0]

    tmp_weather_db.insert_generated_daily_data([
        (job_id, "2021-01-01", 5.0, 15.0, 10.0, 0.0, 10.0, 20.0, "N", 40, 70, 55, 1010.0, 1020.0),
    ])
    tmp_weather_db.insert_generated_hourly_data([
        (job_id, "2021-01-01T00:00:00Z", 8.0, 0.0, 10.0, "N", 55, 1015.0),
        (job_id, "2021-01-01T01:00:00Z", 7.5, 0.0, 9.0, "N", 56, 1015.0),
    ])

    hourly = tmp_weather_db.get_generated_hourly_data(job_id)

    assert len(hourly) == 2
    assert hourly[0]["datetime"] == "2021-01-01T00:00:00Z"


def test_clear_all_data_empties_tables(tmp_weather_db):
    job_id = tmp_weather_db.insert_generation_jobs([
        (40.0, -3.0, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ])[0]
    tmp_weather_db.insert_monthly_predictions([
        (job_id, 2021, 1, "precipitation", 0.0, 2.0, 5.0),
    ])

    deleted = tmp_weather_db.clear_all_data(reset_sequences=True)

    assert deleted["GenerationJob"] == 1
    assert deleted["MonthlyPredictions"] == 1
    assert tmp_weather_db.get_generation_job_info(job_id) is None
