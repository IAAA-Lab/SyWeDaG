import sqlite3 as sql
from utils.system_utils import get_resource_path

def get_db_path():
    """Get database path"""
    return get_resource_path("data/weather.db")

def createDB():
    """Create database file if it doesn't exist"""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sql.connect(str(db_path))
    conn.commit()
    conn.close()

def createTables():
    """Create all tables with schema"""
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    # ========================================================================
    # TABLE 1: WeatherStations
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS WeatherStations (
            source TEXT NOT NULL,
            id_station INTEGER NOT NULL,
            name TEXT NOT NULL,
            region TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            height INTEGER NOT NULL,
            
            PRIMARY KEY (source, id_station)
        )
    ''')
    
    # ========================================================================
    # TABLE 2: HistoricalDataDaily
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS HistoricalDataDaily (
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            id_station INTEGER NOT NULL,
            
            temperature_min REAL,
            temperature_max REAL,
            temperature_mean REAL,
            hour_tmin TEXT,
            hour_tmax TEXT,
            
            precipitation REAL,
            
            wind_speed_mean REAL,
            wind_speed_max REAL,
            wind_direction TEXT CHECK(wind_direction IS NULL OR wind_direction IN ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')),
            hour_wind_max TEXT,
            
            humidity_min INTEGER,
            humidity_max INTEGER,
            humidity_mean INTEGER,
            hour_hrmin TEXT,
            hour_hrmax TEXT,
            
            pressure_min REAL,
            pressure_max REAL,
            hour_presmin TEXT,
            hour_presmax TEXT,
            
            PRIMARY KEY (date, source, id_station),
            FOREIGN KEY (source, id_station) REFERENCES WeatherStations(source, id_station) ON DELETE CASCADE
        )
    ''')
    
    # ========================================================================
    # TABLE 3: GenerationJob
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS GenerationJob (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            historicalStartDate TEXT NOT NULL,
            historicalEndDate TEXT NOT NULL,
            generatedStartDate TEXT NOT NULL,
            generatedEndDate TEXT NOT NULL
        )
    ''')
    
    # ========================================================================
    # TABLE 4: Used_in (Junction table)
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS Used_in (
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            id_station INTEGER NOT NULL,
            idGenerationJob INTEGER NOT NULL,
            
            PRIMARY KEY (date, source, id_station, idGenerationJob),
            FOREIGN KEY (date, source, id_station) 
                REFERENCES HistoricalDataDaily(date, source, id_station) ON DELETE CASCADE,
            FOREIGN KEY (idGenerationJob) 
                REFERENCES GenerationJob(id) ON DELETE CASCADE
        )
    ''')
    
    # ========================================================================
    # TABLE 5: MonthlyPredictions (replaces Modification)
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS MonthlyPredictions (
            idGenerationJob INTEGER NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            variable TEXT NOT NULL CHECK(variable IN (
                'precipitation', 'temperature_max', 'temperature_mean', 'temperature_min', 'number_days_rain'
            )),
            minimum REAL,
            mean REAL,
            maximum REAL,
            
            PRIMARY KEY (idGenerationJob, year, month, variable),
            FOREIGN KEY (idGenerationJob) REFERENCES GenerationJob(id) ON DELETE CASCADE
        )
    ''')
    
    # ========================================================================
    # TABLE 6: GeneratedDataDaily
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS GeneratedDataDaily (
            idGenerationJob INTEGER NOT NULL,
            date TEXT NOT NULL,
            
            temperature_min REAL,
            temperature_max REAL,
            temperature_mean REAL,
            
            precipitation REAL,
            
            wind_speed_mean REAL,
            wind_speed_max REAL,
            wind_direction TEXT CHECK(wind_direction IS NULL OR wind_direction IN ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')),
            
            humidity_min INTEGER,
            humidity_max INTEGER,
            humidity_mean INTEGER,
            
            pressure_min REAL,
            pressure_max REAL,
            
            PRIMARY KEY (idGenerationJob, date),
            FOREIGN KEY (idGenerationJob) REFERENCES GenerationJob(id) ON DELETE CASCADE
        )
    ''')
    
    # ========================================================================
    # TABLE 7: GeneratedDataHourly
    # ========================================================================
    c.execute('''
        CREATE TABLE IF NOT EXISTS GeneratedDataHourly (
            idGenerationJob INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            temperature REAL,
            precipitation REAL,
            wind_speed REAL,
            wind_direction TEXT CHECK(wind_direction IN ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')),
            humidity INTEGER,
            pressure REAL,
            
            PRIMARY KEY (idGenerationJob, datetime),
            FOREIGN KEY (idGenerationJob) REFERENCES GenerationJob(id) ON DELETE CASCADE
        )
    ''')
    
    # ========================================================================
    # CREATE INDEXES
    # ========================================================================
    # HistoricalDataDaily indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_historical_date ON HistoricalDataDaily(date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_historical_station ON HistoricalDataDaily(source, id_station)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_historical_location ON HistoricalDataDaily(source, id_station, date)')
    
    # GenerationJob indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_job_dates ON GenerationJob(generatedStartDate, generatedEndDate)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_job_location ON GenerationJob(latitude, longitude)')
    
    # MonthlyPredictions indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_job ON MonthlyPredictions(idGenerationJob)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_period ON MonthlyPredictions(year, month)')
    
    # GeneratedDataDaily indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_gen_daily_job ON GeneratedDataDaily(idGenerationJob)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_gen_daily_date ON GeneratedDataDaily(date)')
    
    # GeneratedDataHourly indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_gen_hourly_job ON GeneratedDataHourly(idGenerationJob)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_gen_hourly_datetime ON GeneratedDataHourly(datetime)')
    
    # WeatherStations indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_station_location ON WeatherStations(latitude, longitude)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_station_source ON WeatherStations(source)')
    
    conn.commit()
    conn.close()

def init_database():
    """Initialize database (create if not exists)"""
    createDB()
    createTables()

# ============================================================================
# INSERT FUNCTIONS
# ============================================================================

def insert_weather_stations(stations_data):
    """
    Insert multiple weather stations
    
    Args:
        stations_data: List of tuples (source, id_station, name, region, latitude, longitude, height)
    
    Returns:
        Number of rows inserted
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.executemany('''
        INSERT OR REPLACE INTO WeatherStations 
        (source, id_station, name, region, latitude, longitude, height)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', stations_data)
    
    rows_inserted = c.rowcount
    conn.commit()
    conn.close()
    return rows_inserted

def insert_historical_daily_data(daily_data):
    """
    Insert multiple daily historical data records
    
    Args:
        daily_data: List of tuples (date, source, id_station, temperature_min, temperature_max,
                    temperature_mean, hour_tmin, hour_tmax, precipitation, wind_speed_mean,
                    wind_speed_max, wind_direction, hour_wind_max, humidity_min, humidity_max,
                    humidity_mean, hour_hrmin, hour_hrmax, pressure_min, pressure_max,
                    hour_presmin, hour_presmax)
    
    Returns:
        Number of rows inserted
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.executemany('''
        INSERT OR REPLACE INTO HistoricalDataDaily 
        (date, source, id_station, temperature_min, temperature_max, temperature_mean,
         hour_tmin, hour_tmax, precipitation, wind_speed_mean, wind_speed_max,
         wind_direction, hour_wind_max, humidity_min, humidity_max, humidity_mean,
         hour_hrmin, hour_hrmax, pressure_min, pressure_max, hour_presmin, hour_presmax)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', daily_data)
    
    rows_inserted = c.rowcount
    conn.commit()
    conn.close()
    return rows_inserted

def insert_generation_jobs(jobs_data):
    """
    Insert multiple generation jobs
    
    Args:
        jobs_data: List of tuples (latitude, longitude, historicalStartDate, 
                   historicalEndDate, generatedStartDate, generatedEndDate)
    
    Returns:
        List of inserted job IDs
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    inserted_ids = []
    for job in jobs_data:
        c.execute('''
            INSERT INTO GenerationJob 
            (latitude, longitude, historicalStartDate, historicalEndDate, 
             generatedStartDate, generatedEndDate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', job)
        inserted_ids.append(c.lastrowid)
    
    conn.commit()
    conn.close()
    return inserted_ids

def insert_used_in(used_in_data):
    """
    Insert multiple used_in junction records
    
    Args:
        used_in_data: List of tuples (date, source, id_station, idGenerationJob)
    
    Returns:
        Number of rows inserted
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.executemany('''
        INSERT OR REPLACE INTO Used_in 
        (date, source, id_station, idGenerationJob)
        VALUES (?, ?, ?, ?)
    ''', used_in_data)
    
    rows_inserted = c.rowcount
    conn.commit()
    conn.close()
    return rows_inserted

def insert_monthly_predictions(predictions_data):
    """
    Insert multiple monthly predictions (from Excel upload)
    
    Args:
        predictions_data: List of tuples (idGenerationJob, year, month, variable, 
                         minimum, mean, maximum)
    
    Returns:
        Number of rows inserted
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.executemany('''
        INSERT OR REPLACE INTO MonthlyPredictions 
        (idGenerationJob, year, month, variable, minimum, mean, maximum)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', predictions_data)
    
    rows_inserted = c.rowcount
    conn.commit()
    conn.close()
    return rows_inserted

def insert_generated_daily_data(generated_daily):
    """
    Insert multiple generated daily data records
    
    Args:
        generated_daily: List of tuples (idGenerationJob, date, temperature_min,
                        temperature_max, temperature_mean, precipitation, wind_speed_mean,
                        wind_speed_max, wind_direction, humidity_min, humidity_max,
                        humidity_mean, pressure_min, pressure_max)
    
    Returns:
        Number of rows inserted
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.executemany('''
        INSERT OR REPLACE INTO GeneratedDataDaily 
        (idGenerationJob, date, temperature_min, temperature_max, temperature_mean,
         precipitation, wind_speed_mean, wind_speed_max, wind_direction,
         humidity_min, humidity_max, humidity_mean, pressure_min, pressure_max)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', generated_daily)
    
    rows_inserted = c.rowcount
    conn.commit()
    conn.close()
    return rows_inserted

def insert_generated_hourly_data(generated_hourly):
    """
    Insert multiple generated hourly data records
    
    Args:
        generated_hourly: List of tuples (idGenerationJob, datetime, temperature, 
                         precipitation, wind_speed, wind_direction, humidity, pressure)
    
    Returns:
        Number of rows inserted
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.executemany('''
        INSERT OR REPLACE INTO GeneratedDataHourly 
        (idGenerationJob, datetime, temperature, precipitation, wind_speed, 
         wind_direction, humidity, pressure)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', generated_hourly)
    
    rows_inserted = c.rowcount
    conn.commit()
    conn.close()
    return rows_inserted

# ============================================================================
# QUERY FUNCTIONS
# ============================================================================

def get_historical_daily_data(source: str, id_station, start_date: str, end_date: str):
    """
    Recupera datos históricos diarios de una estación entre dos fechas.
    
    Args:
        source: Fuente de datos (ej: 'AEMET')
        id_station: ID de la estación
        start_date: Fecha de inicio (formato 'YYYY-MM-DD')
        end_date: Fecha de fin (formato 'YYYY-MM-DD')
    
    Returns:
        Lista de diccionarios con los datos históricos diarios
    """
    # Normalizar fechas (quitar parte horaria si viene)
    start_date = start_date[:10]
    end_date = end_date[:10]
    
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.execute('''
        SELECT date, temperature_min, temperature_max, temperature_mean,
               hour_tmin, hour_tmax, precipitation, wind_speed_mean, wind_speed_max,
               wind_direction, hour_wind_max, humidity_min, humidity_max, humidity_mean,
               hour_hrmin, hour_hrmax, pressure_min, pressure_max,
               hour_presmin, hour_presmax
        FROM HistoricalDataDaily
        WHERE source = ? 
          AND id_station = ?
          AND date >= ?
          AND date <= ?
        ORDER BY date ASC
    ''', (source, id_station, start_date, end_date))
    
    rows = c.fetchall()
    conn.close()
    
    daily_data = []
    for row in rows:
        daily_data.append({
            'date': row[0],
            'temperature_min': row[1],
            'temperature_max': row[2],
            'temperature_mean': row[3],
            'hour_tmin': row[4],
            'hour_tmax': row[5],
            'precipitation': row[6],
            'wind_speed_mean': row[7],
            'wind_speed_max': row[8],
            'wind_direction': row[9],
            'hour_wind_max': row[10],
            'humidity_min': row[11],
            'humidity_max': row[12],
            'humidity_mean': row[13],
            'hour_hrmin': row[14],
            'hour_hrmax': row[15],
            'pressure_min': row[16],
            'pressure_max': row[17],
            'hour_presmin': row[18],
            'hour_presmax': row[19]
        })
    
    return daily_data

def get_generated_hourly_data(job_id: int):
    """
    Recupera todos los datos horarios generados para un job específico.
    
    Args:
        job_id: ID del trabajo de generación
    
    Returns:
        Lista de diccionarios con los datos horarios generados
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.execute('''
        SELECT datetime, temperature, precipitation, wind_speed, 
               wind_direction, humidity, pressure
        FROM GeneratedDataHourly
        WHERE idGenerationJob = ?
        ORDER BY datetime ASC
    ''', (job_id,))
    
    rows = c.fetchall()
    conn.close()
    
    hourly_data = []
    for row in rows:
        hourly_data.append({
            'datetime': row[0],
            'temperature': row[1],
            'precipitation': row[2],
            'wind_speed': row[3],
            'wind_direction': row[4],
            'humidity': row[5],
            'pressure': row[6]
        })
    
    return hourly_data

def get_monthly_predictions(job_id: int):
    """
    Recupera las predicciones mensuales para un job.
    
    Args:
        job_id: ID del trabajo de generación
    
    Returns:
        Lista de diccionarios con las predicciones mensuales
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.execute('''
        SELECT year, month, variable, minimum, mean, maximum
        FROM MonthlyPredictions
        WHERE idGenerationJob = ?
        ORDER BY year ASC, month ASC, variable ASC
    ''', (job_id,))
    
    rows = c.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        predictions.append({
            'year': row[0],
            'month': row[1],
            'variable': row[2],
            'minimum': row[3],
            'mean': row[4],
            'maximum': row[5]
        })
    
    return predictions

def get_generation_job_info(job_id: int):
    """
    Recupera la información del job de generación.
    
    Args:
        job_id: ID del trabajo de generación
    
    Returns:
        Diccionario con la información del job o None si no existe
    """
    db_path = get_db_path()
    conn = sql.connect(str(db_path))
    c = conn.cursor()
    
    c.execute('''
        SELECT latitude, longitude, historicalStartDate, historicalEndDate, 
               generatedStartDate, generatedEndDate
        FROM GenerationJob
        WHERE id = ?
    ''', (job_id,))
    
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'latitude': row[0],
            'longitude': row[1],
            'historicalStartDate': row[2],
            'historicalEndDate': row[3],
            'generatedStartDate': row[4],
            'generatedEndDate': row[5]
        }
    return None

if __name__ == "__main__":
    init_database()
    print("✅ Database initialized successfully")

