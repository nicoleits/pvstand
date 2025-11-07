import clickhouse_connect
import pandas as pd

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': 30091,
    'username': "default",
    'password': "Psda2020"
}

# Conexión al servidor
client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)

# Consulta SQL para el 04-11-2025
query = """
SELECT
    timestamp,
    `1TE418(C)` AS "1TE418(C)",
    `1TE419(C)` AS "1TE419(C)"
FROM PSDA.fixed_plant_atamo_1
WHERE toDate(timestamp) = '2025-11-04'
ORDER BY timestamp ASC
"""

# Ejecutar consulta y convertir a DataFrame
result = client.query(query)
df = pd.DataFrame(result.result_rows, columns=result.column_names)

# Mostrar primeras filas
print(df.head())

# Guardar en CSV
df.to_csv("pvstand/datos/data_temp.csv", index=False)
print("Datos guardados en data_temp.csv")