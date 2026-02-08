"""
Módulo de Inferencia (IA) - Degradación de Baterías
===================================================
Carga una base de datos de análisis masivo y predice la degradación
basándose en "Gemelos Digitales" (KNN).

NOVEDAD V2.1: Normalización de Resistencia
------------------------------------------
La resistencia interna escala inversamente a la capacidad (pilas grandes tienen menos R).
Para extrapolar datos de laboratorio (celdas pequeñas) a tu pack (celdas grandes),
calculamos una métrica agnóstica del tamaño:

    Métrica = Caída de Voltaje Normalizada por Ciclo (a 1C)
    
Luego, en aging.py, se des-normaliza:
    Delta_R (Ohm) = Métrica / Capacidad_Tu_Celda (Ah)
"""

import pandas as pd
import numpy as np
import os

class CerebroDegradacion:
    def __init__(self, archivo_csv=None):
        """
        Inicializa el módulo de inferencia.
        Si no se pasa ruta, busca automáticamente en 'data/Resultado_Analisis_Bateria.csv'
        """
        self.archivo_csv = self._encontrar_archivo(archivo_csv)
        self.df = pd.DataFrame()
        self.datos_cargados = False
        self._cargar_y_reparar_datos()

    def _encontrar_archivo(self, ruta_usuario):
        # 1. Si el usuario da una ruta y existe, usarla
        if ruta_usuario and os.path.exists(ruta_usuario):
            return ruta_usuario
            
        # 2. Rutas candidatas automáticas (Prioridad: carpeta data/)
        candidatos = [
            os.path.join("data", "Resultado_Analisis_Bateria.csv"),  # Estándar
            "Resultado_Analisis_Bateria.csv",                        # Raíz
            os.path.join(os.path.dirname(__file__), "..", "data", "Resultado_Analisis_Bateria.csv"), # Relativo
            os.path.join(os.path.dirname(__file__), "data", "Resultado_Analisis_Bateria.csv"),
            r"C:\Temp_Analisis\Bateria_Lab_Project\data\Resultado_Analisis_Bateria.csv"
        ]
        
        for ruta in candidatos:
            if os.path.exists(ruta):
                return ruta
                
        return "data/Resultado_Analisis_Bateria.csv" # Fallback

    def _cargar_y_reparar_datos(self):
        if not os.path.exists(self.archivo_csv):
            print(f"[IA] ⚠️ AVISO: No encuentro la base de datos en: '{self.archivo_csv}'")
            return

        try:
            # 1. Cargar el CSV
            self.df = pd.read_csv(self.archivo_csv)

            # 2. Validar columnas críticas
            cols_req = ['C_rate', 'Cap_Max', 'Slope_Capacity', 'Slope_EODV']
            for col in cols_req:
                if col not in self.df.columns:
                    print(f"[IA] ❌ Error: Falta columna '{col}' en el CSV.")
                    return
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Eliminar filas corruptas
            self.df = self.df.dropna(subset=cols_req)
            
            # 3. --- CÁLCULO DE MÉTRICA DE RESISTENCIA NORMALIZADA ---
            # Problema: R (Ohms) depende del tamaño de la celda (1/Capacidad).
            # Solución: Usar "Caída de Voltaje Normalizada".
            # Derivación:
            #   Delta_V = I * Delta_R
            #   Delta_R = Delta_V / I
            #   I = C_rate * Capacidad
            #   Delta_R = Delta_V / (C_rate * Capacidad)
            #
            #   Queremos una métrica 'M' tal que: Delta_R_usuario = M / Capacidad_usuario
            #   Por tanto: M = Delta_R * Capacidad = Delta_V / C_rate
            
            # Evitar división por cero si C_rate es muy bajo
            c_rates_seguros = self.df['C_rate'].replace(0, 0.1)
            
            # Métrica = Voltios perdidos por ciclo si descargáramos a 1C
            # Unidades: [V / ciclo] (independiente de Ah)
            self.df['voltage_drop_norm_per_cycle'] = self.df['Slope_EODV'].abs() / c_rates_seguros
            
            # 4. Normalizar degradación de capacidad (%/ciclo)
            self.df['capacity_fade_pct'] = self.df['Slope_Capacity'].abs() / self.df['Cap_Max']
            
            # 5. Filtros de Calidad
            # Descartamos celdas con degradación absurda (ruido de medición)
            # Umbral: > 5mV por ciclo normalizado es probablemente fallo de test, no degradación natural
            self.df = self.df[self.df['voltage_drop_norm_per_cycle'] < 0.010] 
            self.df = self.df[self.df['capacity_fade_pct'] < 0.005]

            print(f"[IA] ✅ Cerebro cargado: {len(self.df)} perfiles.")
            print(f"     - Deg. Cap Media: {self.df['capacity_fade_pct'].mean():.5%}/ciclo")
            print(f"     - Métrica Res. Media: {self.df['voltage_drop_norm_per_cycle'].mean():.6f} V_norm/ciclo")
            
            self.datos_cargados = True

        except Exception as e:
            print(f"[IA] ❌ ERROR CRÍTICO leyendo CSV: {e}")

    def predecir_degradacion(self, c_rate_usuario, temperatura_usuario=25.0, dod_usuario=100.0):
        """
        Retorna las tasas BASE de envejecimiento.
        
        Args:
            c_rate_usuario (float): Tasa de descarga efectiva.
            
        Returns:
            deg_cap_base (float): % Capacidad perdida por ciclo.
            info (str): Texto explicativo.
        """
        if not self.datos_cargados or self.df.empty:
            # Fallback seguro: valores conservadores
            return 0.00015, "Modo Fallback (Sin datos)"

        # 1. Buscar Gemelos (KNN por C-rate)
        # Buscamos experimentos que se hicieran a una velocidad similar
        # Optimización: Usamos NumPy directo para evitar overhead de pandas y sorting completo
        c_rates = self.df['C_rate'].values
        distancias = np.abs(c_rates - c_rate_usuario)

        # Encontrar los k índices con menor distancia
        k = 5
        if len(distancias) > k:
            # argpartition pone los k menores al principio (sin orden garantizado entre ellos)
            # Esto es O(n) vs O(n log n) de sort
            idx = np.argpartition(distancias, k)[:k]
        else:
            # Fallback para datasets diminutos
            idx = np.arange(len(distancias))

        vecinos = self.df.iloc[idx]

        # 2. Promediar comportamiento
        deg_cap_base = vecinos['capacity_fade_pct'].mean()
        metric_res_norm = vecinos['voltage_drop_norm_per_cycle'].mean()

        # Info para el usuario
        info = (f"Gemelos: {len(vecinos)} (C-rate medio: {vecinos['C_rate'].mean():.1f}C)")

        return deg_cap_base, info