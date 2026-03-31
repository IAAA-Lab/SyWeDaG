from datetime import datetime, timedelta
from typing import List

from data_sources.base_source import DailyWeatherRecord, WeatherData
from utils.data_parsing import parse_float, parse_int

# Functions for filling missing days and interpolating/extrapolating values in historical weather data.
def fill_missing_days(daily_records: List[dict]) -> List[dict]:
    """
    Fill missing days by interpolating between available days or extrapolating at extremes.

    If there are previous and next days, interpolates between them.
    If only previous day exists (e.g. final extreme), extrapolates using that day.
    If only next day exists (e.g. initial extreme), extrapolates using that day.

    Args:
        daily_records: List of daily records (may have missing days)

    Returns:
        Complete list of daily records without gaps
    """
    if not daily_records:
        return []

    daily_records = sorted(daily_records, key=lambda x: x.get('date', ''))
    records_by_date = {record['date']: record for record in daily_records}

    start_date = datetime.strptime(daily_records[0]['date'], '%Y-%m-%d').date()
    end_date = datetime.strptime(daily_records[-1]['date'], '%Y-%m-%d').date()

    complete_records = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')

        if date_str in records_by_date:
            complete_records.append(records_by_date[date_str])
        else:
            prev_date = current_date - timedelta(days=1)
            next_date = current_date + timedelta(days=1)

            while prev_date >= start_date and prev_date.strftime('%Y-%m-%d') not in records_by_date:
                prev_date -= timedelta(days=1)

            while next_date <= end_date and next_date.strftime('%Y-%m-%d') not in records_by_date:
                next_date += timedelta(days=1)

            has_prev = prev_date >= start_date
            has_next = next_date <= end_date

            if has_prev and has_next:
                prev_record = records_by_date[prev_date.strftime('%Y-%m-%d')]
                next_record = records_by_date[next_date.strftime('%Y-%m-%d')]
                filled_record = _interpolate_daily_record(prev_record, next_record, date_str)
                complete_records.append(filled_record)
            elif has_prev:
                prev_record = records_by_date[prev_date.strftime('%Y-%m-%d')]
                filled_record = _extrapolate_daily_record(prev_record, date_str)
                complete_records.append(filled_record)
            elif has_next:
                next_record = records_by_date[next_date.strftime('%Y-%m-%d')]
                filled_record = _extrapolate_daily_record(next_record, date_str)
                complete_records.append(filled_record)

        current_date += timedelta(days=1)

    return complete_records


def _extrapolate_daily_record(reference_record: dict, target_date: str) -> dict:
    """
    Extrapolate a daily record using a reference one when days are missing at extremes.

    When there are no previous or next days available (e.g. 1-2 January, 31 December),
    uses the values from the nearest available day as reference for the missing day.

    Args:
        reference_record: Record from the nearest known day
        target_date: Date of the day to extrapolate (format YYYY-MM-DD)

    Returns:
        Extrapolated daily record with values similar to the reference
    """
    extrapolated = reference_record.copy()
    extrapolated['date'] = target_date
    return extrapolated


def _interpolate_daily_record(prev_record: dict, next_record: dict, target_date: str) -> dict:
    """
    Linearly interpolate a daily record between two known records.

    For each variable:
    - If both records have values: returns the average
    - If only one has a value: returns that value
    - If neither has a value: returns None

    Args:
        prev_record: Previous day record
        next_record: Next day record
        target_date: Date of the day to interpolate (format YYYY-MM-DD)

    Returns:
        Interpolated daily record
    """

    def interpolate_value(key, parse_as='float'):
        prev_val_raw = prev_record.get(key)
        next_val_raw = next_record.get(key)

        if parse_as == 'float':
            prev_val = parse_float(prev_val_raw)
            next_val = parse_float(next_val_raw)

            if prev_val is not None and next_val is not None:
                return str(round((prev_val + next_val) / 2, 1))
            if prev_val is not None:
                return str(round(prev_val, 1))
            if next_val is not None:
                return str(round(next_val, 1))
            return None

        if parse_as == 'int':
            prev_val = parse_int(prev_val_raw)
            next_val = parse_int(next_val_raw)

            if prev_val is not None and next_val is not None:
                return str(int((prev_val + next_val) / 2))
            if prev_val is not None:
                return str(prev_val)
            if next_val is not None:
                return str(next_val)
            return None

        if prev_val_raw is not None and prev_val_raw != '':
            return prev_val_raw
        if next_val_raw is not None and next_val_raw != '':
            return next_val_raw
        return None

    interpolated_record = {
        'date': target_date,
        'station_id': prev_record.get('station_id') or next_record.get('station_id'),
        'station_name': prev_record.get('station_name') or next_record.get('station_name'),
        'province': prev_record.get('province') or next_record.get('province'),
        'altitude': prev_record.get('altitude') or next_record.get('altitude'),
    }

    interpolated_record['temperature_mean'] = interpolate_value('temperature_mean', 'float')
    interpolated_record['temperature_min'] = interpolate_value('temperature_min', 'float')
    interpolated_record['temperature_max'] = interpolate_value('temperature_max', 'float')

    interpolated_record['hour_tmin'] = interpolate_value('hour_tmin', 'string')
    interpolated_record['hour_tmax'] = interpolate_value('hour_tmax', 'string')

    interpolated_record['precipitation'] = interpolate_value('precipitation', 'float')

    interpolated_record['wind_direction'] = interpolate_value('wind_direction', 'string')
    interpolated_record['wind_speed_mean'] = interpolate_value('wind_speed_mean', 'float')
    interpolated_record['wind_speed_max'] = interpolate_value('wind_speed_max', 'float')
    interpolated_record['hour_wind_max'] = interpolate_value('hour_wind_max', 'string')

    interpolated_record['humidity_mean'] = interpolate_value('humidity_mean', 'int')
    interpolated_record['humidity_min'] = interpolate_value('humidity_min', 'int')
    interpolated_record['humidity_max'] = interpolate_value('humidity_max', 'int')
    interpolated_record['hour_hrmin'] = interpolate_value('hour_hrmin', 'string')
    interpolated_record['hour_hrmax'] = interpolate_value('hour_hrmax', 'string')

    interpolated_record['pressure_min'] = interpolate_value('pressure_min', 'float')
    interpolated_record['pressure_max'] = interpolate_value('pressure_max', 'float')
    interpolated_record['hour_presmin'] = interpolate_value('hour_presmin', 'string')
    interpolated_record['hour_presmax'] = interpolate_value('hour_presmax', 'string')

    return interpolated_record


def interpolate_missing_values_in_period(daily_records: List[dict]) -> None:
    """
    Interpolate missing values (NULLs) within existing days.

    Detect which variables are never returned (so they are not interpolated),
    and for variables that are returned but contain NULLs,
    interpolate those values using nearby days.

    If there are previous and next days, interpolate between them.
    If only a previous day exists, extrapolate forward using that day.
    If only a next day exists, extrapolate backward using that day.
    If there are no nearby values, NULL is preserved.

    It also interpolates hour fields and wind direction.

    Args:
        daily_records: List of daily records (modified in-place)
    """
    if not daily_records or len(daily_records) < 2:
        return

    numeric_vars_keys = {
        'temperature_min': ['temperature_min'],
        'temperature_max': ['temperature_max'],
        'temperature_mean': ['temperature_mean'],
        'precipitation': ['precipitation'],
        'wind_speed_mean': ['wind_speed_mean'],
        'wind_speed_max': ['wind_speed_max'],
        'humidity_min': ['humidity_min'],
        'humidity_max': ['humidity_max'],
        'humidity_mean': ['humidity_mean'],
        'pressure_min': ['pressure_min'],
        'pressure_max': ['pressure_max'],
    }

    string_vars_keys = {
        'hour_tmin': ['hour_tmin'],
        'hour_tmax': ['hour_tmax'],
        'hour_wind_max': ['hour_wind_max'],
        'hour_hrmin': ['hour_hrmin'],
        'hour_hrmax': ['hour_hrmax'],
        'hour_presmin': ['hour_presmin'],
        'hour_presmax': ['hour_presmax'],
        'wind_direction': ['wind_direction'],
    }

    actual_keys_used = {}
    for var, possible_keys in numeric_vars_keys.items():
        actual_key = None
        for record in daily_records:
            for key in possible_keys:
                if key in record:
                    actual_key = key
                    break
            if actual_key:
                break
        if actual_key:
            actual_keys_used[var] = actual_key

    actual_string_keys_used = {}
    for var, possible_keys in string_vars_keys.items():
        actual_key = None
        for record in daily_records:
            for key in possible_keys:
                if key in record:
                    actual_key = key
                    break
            if actual_key:
                break
        if actual_key:
            actual_string_keys_used[var] = actual_key

    never_present = set()
    for var, actual_key in actual_keys_used.items():
        has_any_value = False
        for record in daily_records:
            val = record.get(actual_key)
            if val is not None and val != '' and parse_float(val) is not None:
                has_any_value = True
                break
        if not has_any_value:
            never_present.add(var)

    if never_present:
        print(f"⚠️ Variables without data in period (not interpolating): {', '.join(sorted(never_present))}")

    for var, actual_key in actual_keys_used.items():
        if var in never_present:
            continue

        for i in range(len(daily_records)):
            val = daily_records[i].get(actual_key)
            parsed_val = parse_float(val)

            if parsed_val is None:
                prev_val = None
                next_val = None

                for j in range(i - 1, -1, -1):
                    prev_parsed = parse_float(daily_records[j].get(actual_key))
                    if prev_parsed is not None:
                        prev_val = prev_parsed
                        break

                for j in range(i + 1, len(daily_records)):
                    next_parsed = parse_float(daily_records[j].get(actual_key))
                    if next_parsed is not None:
                        next_val = next_parsed
                        break

                if prev_val is not None and next_val is not None:
                    interpolated = (prev_val + next_val) / 2
                    daily_records[i][actual_key] = str(round(interpolated, 1))
                elif prev_val is not None:
                    daily_records[i][actual_key] = str(prev_val)
                elif next_val is not None:
                    daily_records[i][actual_key] = str(next_val)

    for _, actual_key in actual_string_keys_used.items():
        for i in range(len(daily_records)):
            val = daily_records[i].get(actual_key)
            is_invalid = val is None or val == '' or str(val).lower() in ['varias', 'multiple', 'n/a', 'nd', 'ind', 'vv']

            if is_invalid:
                prev_val = None
                next_val = None

                for j in range(i - 1, -1, -1):
                    prev_candidate = daily_records[j].get(actual_key)
                    if prev_candidate and prev_candidate != '' and str(prev_candidate).lower() not in ['varias', 'multiple', 'n/a', 'nd', 'ind', 'vv']:
                        prev_val = prev_candidate
                        break

                for j in range(i + 1, len(daily_records)):
                    next_candidate = daily_records[j].get(actual_key)
                    if next_candidate and next_candidate != '' and str(next_candidate).lower() not in ['varias', 'multiple', 'n/a', 'nd', 'ind', 'vv']:
                        next_val = next_candidate
                        break

                if prev_val is not None:
                    daily_records[i][actual_key] = prev_val
                elif next_val is not None:
                    daily_records[i][actual_key] = next_val

# Functions for checking if historical treatment is needed and applying it to weather data.
def _needs_historical_treatment(daily_records: List[DailyWeatherRecord]) -> bool:
    if len(daily_records) < 2:
        return False

    sorted_records = sorted(daily_records, key=lambda record: record.date)

    for index in range(1, len(sorted_records)):
        prev_date = datetime.strptime(sorted_records[index - 1].date, '%Y-%m-%d').date()
        curr_date = datetime.strptime(sorted_records[index].date, '%Y-%m-%d').date()
        if (curr_date - prev_date).days > 1:
            return True

    for record in sorted_records:
        if (
            record.temperature_min is None
            or record.temperature_max is None
            or record.temperature_mean is None
            or record.precipitation is None
            or record.wind_speed_mean is None
            or record.wind_speed_max is None
            or record.wind_direction is None
            or record.humidity_min is None
            or record.humidity_max is None
            or record.humidity_mean is None
            or record.pressure_min is None
            or record.pressure_max is None
        ):
            return True

    return False


def _convert_weather_data_to_treatment_records(weather_data: WeatherData) -> List[dict]:
    treatment_records = []
    for record in weather_data.daily_records:
        treatment_records.append(
            {
                'date': record.date,
                'temperature_min': record.temperature_min,
                'temperature_max': record.temperature_max,
                'temperature_mean': record.temperature_mean,
                'hour_tmin': record.hour_tmin,
                'hour_tmax': record.hour_tmax,
                'precipitation': record.precipitation,
                'wind_speed_mean': record.wind_speed_mean,
                'wind_speed_max': record.wind_speed_max,
                'wind_direction': record.wind_direction,
                'hour_wind_max': record.hour_wind_max,
                'humidity_min': record.humidity_min,
                'humidity_max': record.humidity_max,
                'humidity_mean': record.humidity_mean,
                'hour_hrmin': record.hour_hrmin,
                'hour_hrmax': record.hour_hrmax,
                'pressure_min': record.pressure_min,
                'pressure_max': record.pressure_max,
                'hour_presmin': record.hour_presmin,
                'hour_presmax': record.hour_presmax,
            }
        )
    return treatment_records


def _convert_treatment_records_to_weather_data(treatment_records: List[dict]) -> WeatherData:
    daily_records = []
    for daily_record in treatment_records:
        record = DailyWeatherRecord(
            date=daily_record.get('date', ''),
            temperature_min=parse_float(daily_record.get('temperature_min')),
            temperature_max=parse_float(daily_record.get('temperature_max')),
            temperature_mean=parse_float(daily_record.get('temperature_mean')),
            hour_tmin=daily_record.get('hour_tmin'),
            hour_tmax=daily_record.get('hour_tmax'),
            precipitation=parse_float(daily_record.get('precipitation')),
            wind_speed_mean=parse_float(daily_record.get('wind_speed_mean')),
            wind_speed_max=parse_float(daily_record.get('wind_speed_max')),
            wind_direction=daily_record.get('wind_direction'),
            hour_wind_max=daily_record.get('hour_wind_max'),
            humidity_min=parse_int(daily_record.get('humidity_min')),
            humidity_max=parse_int(daily_record.get('humidity_max')),
            humidity_mean=parse_int(daily_record.get('humidity_mean')),
            hour_hrmin=daily_record.get('hour_hrmin'),
            hour_hrmax=daily_record.get('hour_hrmax'),
            pressure_min=parse_float(daily_record.get('pressure_min')),
            pressure_max=parse_float(daily_record.get('pressure_max')),
            hour_presmin=daily_record.get('hour_presmin'),
            hour_presmax=daily_record.get('hour_presmax'),
        )
        daily_records.append(record)

    return WeatherData(daily_records=daily_records)


def apply_historical_treatment_if_needed(weather_data: WeatherData) -> WeatherData:
    if weather_data is None or not weather_data.daily_records:
        return weather_data

    if not _needs_historical_treatment(weather_data.daily_records):
        return weather_data

    print("ℹ️ Applying historical interpolation/extrapolation")
    treatment_records = _convert_weather_data_to_treatment_records(weather_data)
    treatment_records = fill_missing_days(treatment_records)
    interpolate_missing_values_in_period(treatment_records)
    return _convert_treatment_records_to_weather_data(treatment_records)
