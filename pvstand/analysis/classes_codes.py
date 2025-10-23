# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:43:10 2025

@author: Nicole Torres
"""

import pandas as pd
import clickhouse_connect
import os
import ephem
import numpy as np
import datetime
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient, Point, Dialect
from influxdb_client.client.write_api import SYNCHRONOUS

class medio_dia_solar:


    def __init__(self, datei = '2024-12-01', datef = '2025-01-01',freq="1d",
                 inter = 30):
        self.datei = datei
        self.datef = datef
        self.freq = freq
        self.inter = inter

    def msd(self):
        time_s = pd.date_range(start=self.datei,end=self.datef,freq=self.freq)
        lista = []
    
        for date in time_s:
            o = ephem.Observer()
            o.lat, o.long = '-24.0900', '-69.9289'
            sun = ephem.Sun()
            sunrise = o.previous_rising(sun,start=ephem.Date(date + datetime.timedelta(days=1)))
            noon = o.next_transit(sun, start=sunrise)
            sunset = o.next_setting(sun, start=noon)
            lista.append(noon.datetime())
            ephem.date((sunrise + sunset) / 2)
    
        solar_noon = pd.to_datetime(lista)
        solar_noon = pd.to_datetime([i.round("min") for i in solar_noon])
    
    # Rango 30 min antes y 30 min despues del medio dia solar
        time_i = (solar_noon - datetime.timedelta(minutes=self.inter))
        time_f = (solar_noon + datetime.timedelta(minutes=self.inter))
        
        dftime = pd.DataFrame([time_i,time_f]).T
        return dftime

class influxdb_class:

    def __init__(self, token, url, org):
        self.org = org
        self.token = token
        self.url = url

    def query_influxdb(self, bucket, tables, attributes,
                 ts_start, ts_stop=datetime.datetime.now()):
        print(f"Querying InfluxDB: bucket={bucket}, tables={tables}, attributes={attributes}, ts_start={ts_start}, ts_stop={ts_stop}")
        timeout = 1000000

        # Ensure url is a string
        if not isinstance(self.url, str):
            raise ValueError('"url" attribute is not str instance')

        # Define time values
        ts_start = ts_start.strftime("%Y-%m-%dT00:00:00Z")
        ts_stop = ts_stop.strftime("%Y-%m-%dT23:59:59Z")
        window_period = "1m"
        table = " or ".join([f'r["_measurement"] == "{i}"' for i in tables])
        attribute = " or ".join([f'r["_field"] == "{i}"' for i in attributes])

        # Flux query
        query = f'''
        import "date"
        from(bucket: "{bucket}")
          |> range(start: time(v: "{ts_start}"), stop: time(v: {ts_stop}))
          |> filter(fn: (r) => {table})
          |> filter(fn: (r) => {attribute})
          |> aggregateWindow(every: {window_period}, fn: mean, createEmpty: false)
          |> map(fn: (r) => ({{
              r with 
              _value: r._value * 1.0,
              _measurement: r._measurement,
              _time: date.truncate(t: r._time, unit: 1m)
          }}))
          |> yield(name: "mean")
        '''

        print(f"Executing query: {query}")
        # Initialize client
        client = InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=timeout)

        # Execute query
        tables = client.query_api().query(query=query)

        # Process results and store in DataFrame
        data = []
        for table in tables:
            for record in table.records:
                data.append({
                    "StampTime": record["_time"].replace(tzinfo=None),
                    "Module": record["_measurement"],
                    "Attribute": record["_field"],
                    "Value": record["_value"]
                })

        df = pd.DataFrame(data)
        df = df.pivot(index="StampTime", columns=['Module', 'Attribute'], values='Value')
        client.close()
        print(f"Query result: {df.head()}")
        return df

class clickh:

    def __init__(self):
        return

    def iv_manual(self, client_clickhouse,query_clickhouse):
        data_iv_curves = client_clickhouse.query(query_clickhouse)
        print(f"data_iv_curves: {data_iv_curves.result_set[:5]}")

        curves_list = []
        for curve in data_iv_curves.result_set:
            currents = curve[4]
            voltages = curve[3]
            powers = [currents[i] * voltages[i] for i in range(len(currents))]
            timestamp = curve[0]
            module = curve[2]
            pmp = max(powers)
            isc = max(currents)
            voc = max(voltages)
            imp = currents[np.argmax(powers)]
            vmp = voltages[np.argmax(powers)]
            curves_list.append([timestamp, module, pmp, isc, voc, imp, vmp])

        return curves_list
