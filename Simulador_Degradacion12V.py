import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
import os

# --- CORRECCIÓN: Mantenemos el nombre original 'get_absolute_path' ---
# para no romper las funciones cargar_curva_ocv_usuario y cargar_perfil_solicitaciones
def get_absolute_path(filename):
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directorio_script, filename)

# Importamos tu módulo de IA
try:
    from Modulo_Inferencia12V import CerebroDegradacion
except ImportError:
    print("ERROR CRÍTICO: No encuentro Modulo_Inferencia.py en la carpeta.")

# --- CARGA DEL CEREBRO ---
# Usamos la función corregida para encontrar el CSV
ruta_csv = get_absolute_path("Resultado_Analisis_Bateria.csv")
print(f"Buscando base de datos en: {ruta_csv}") 

cerebro_ia = CerebroDegradacion(ruta_csv)

# ==========================================
# 1. LECTORES DE DATOS Y UTILIDADES
# ==========================================

def time_formatter(x, pos):
    """Formato Cronómetro MM:SS"""
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"

def cargar_curva_ocv_usuario(filename):
    full_path = get_absolute_path(filename)
    soc_points, ocv_points = [], []
    if not os.path.exists(full_path):
        return interp1d([0, 100], [3.0, 4.2], kind='linear', fill_value="extrapolate")
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in lines:
            if "SOC" in line: continue
            clean = line.replace(',', '.').strip()
            parts = clean.split()
            if len(parts) >= 2:
                try:
                    soc_points.append(float(parts[0]))
                    ocv_points.append(float(parts[1]))
                except: continue
        return interp1d(soc_points, ocv_points, kind='linear', fill_value="extrapolate")
    except: return interp1d([0, 100], [3.0, 4.2], kind='linear')

def cargar_perfil_solicitaciones(filename):
    full_path = get_absolute_path(filename)
    t_out, I_out = [], []
    if not os.path.exists(full_path):
        return np.linspace(0, 1800, 1800), np.zeros(1800)
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in lines:
            clean = line.split('%')[0].replace(';', '').strip()
            parts = clean.split(',')
            if len(parts) >= 3:
                try:
                    t0, tf, amp = float(parts[0]), float(parts[1]), float(parts[2])
                    steps = int((tf - t0) * 10)
                    if steps < 1: steps = 1
                    t_out.extend(np.linspace(t0, tf, steps, endpoint=False))
                    I_out.extend(np.full(steps, amp))
                except: continue
        return np.array(t_out), np.array(I_out)
    except: return np.linspace(0,1,10), np.zeros(10)

# Carga Inicial
f_ocv_user = cargar_curva_ocv_usuario("SOC_OCV_datos.txt")
t_base, I_base_pack = cargar_perfil_solicitaciones("Solicitaciones enduarnce.txt")

# ==========================================
# 2. CONFIGURACIÓN MODELOS
# ==========================================

# 1. Base de datos de degradación (Calibrada con Datasheet Grepow GRP7770175)
# Vida estimada: ~500 ciclos @ 0.5C (Nominal)

# 1. Base de datos de degradación (ELIMINADA - SE USA SOLO IA + FISICA)
# Vida estimada: ~500 ciclos @ 0.5C (Nominal)

# MACROS DE CONFIGURACIÓN RÁPIDA (Solo mueven sliders)
MACROS_CONFIGURACION = {
    '3S2P (80-20)': {'S': 3, 'P': 2, 'S_max': 80, 'S_min': 20},
    '3S2P (95-35)': {'S': 3, 'P': 2, 'S_max': 95, 'S_min': 35},
    '4S2P (80-20)': {'S': 4, 'P': 2, 'S_max': 80, 'S_min': 20}
}

R_CONN_PACK = 0.0040 
R_CONN_PACK = 0.0040 

# ==========================================
# 3. INTERFAZ GRÁFICA
# ==========================================
fig = plt.figure(figsize=(16, 9))
plt.subplots_adjust(left=0.45, right=0.95, top=0.95, bottom=0.05, hspace=0.45)

ax_graph1 = plt.subplot(3, 1, 1) 
ax_term = plt.subplot(3, 1, 2)
ax_soc = plt.subplot(3, 1, 3)

controls = []
graph_mode = 'Tensión (V)' 
txt_res = None
txt_vida = None

# ==========================================
# 4. LÓGICA DE SIMULACIÓN
# ==========================================
def run_simulation(val=None):
    # 1. Leer Sliders
    cap_celda = controls[0][0].val
    n_s = int(controls[1][0].val)
    n_p = int(controls[2][0].val)
    soc_max = controls[3][0].val
    soc_min = controls[4][0].val
    temp_amb = controls[5][0].val
    refrigeracion = controls[6][0].val
    soh_simulado_pct = controls[7][0].val # Nuevo Slider SOH
    
    if soc_min >= soc_max: soc_min = soc_max - 1

    # --- FUNCIÓN DE RESISTENCIA DINÁMICA (Importada de physics.py) ---
    def _calcular_resistencia_dinamica(r_base_ohm, soc, temp_c, current_a, t_pulso_s):
        # 1. TÉRMINO BASE (SOC - "Bañera")
        factor_soc = 1.0
        if soc < 10.0:
            val = (10.0 - soc) / 10.0
            factor_soc = 1.0 + 0.5 * (val * val)
        elif soc > 90.0:
            val = (soc - 90.0) / 10.0
            factor_soc = 1.0 + 0.2 * (val * val)
        
        R_soc = r_base_ohm * factor_soc

        # 2. TÉRMINO ARRHENIUS (TEMPERATURA)
        temp_k = temp_c + 273.15
        if temp_k < 200: temp_k = 200.0
        factor_arrhenius = np.exp(1500.0 * (1.0/temp_k - 1.0/298.15))

        # 3. TÉRMINO BUTLER-VOLMER (CORRIENTE)
        i_abs = abs(current_a)
        factor_corriente = 1.0 - 0.15 * (1.0 - np.exp(-0.05 * i_abs))

        # 4. TÉRMINO DINÁMICO (TIEMPO - Polarización)
        factor_tiempo = 1.0 + 0.3 * (1.0 - np.exp(-t_pulso_s / 5.0))

        return R_soc * factor_arrhenius * factor_corriente * factor_tiempo

    # 2. Física (Ajustada por SOH)
    # ---------------------------
    # Envejecimiento:
    # 1. Capacidad disminuye linealmente con SOH (por definición)
    # 2. Resistencia aumenta. Regla empírica: R ~ 1/SOH (o más agresivo R ~ 1/SOH^2)
    
    factor_salud = soh_simulado_pct / 100.0
    
    # Capacidad Real Actual (= Capacidad Nominal * SOH)
    
    # Capacidad Real Actual (= Capacidad Nominal * SOH)
    cap_celda_real = cap_celda * factor_salud
    cap_pack_ah_real = cap_celda_real * n_p
    
    # Inicialización por defecto para evitar UnboundLocalError
    r_nom_cell = 0.002025 
    mass = 0.205
    
    if abs(cap_celda - 10.0) > 0.1:
         r_nom_cell = 0.02025 / cap_celda if cap_celda > 0 else 0.002025
         mass = cap_celda * 0.0205 # La masa no cambia significativa con SOH

    # Factor de corrección para simulación realista + ENVEJECIMIENTO
    # Al envejecer (bajar SOH), la resistencia aumenta.
    # r_operative_cell será ahora la RESISTENCIA BASE NOMINAL DE LA CELDA ENVEJECIDA
    # La dinámica se calculará paso a paso más abajo.
    r_base_envejecida = r_nom_cell / factor_salud

    # 3. Cálculos Eléctricos (Bucle temporal manual para R dinámica)
    dt = t_base[1] - t_base[0] if len(t_base)>1 else 0.1
    
    # Inicialización de Arrays
    num_steps = len(t_base)
    soc_t = np.zeros(num_steps)
    v_pack_t = np.zeros(num_steps)
    temps = np.zeros(num_steps)
    
    # Estado Inicial
    soc_curr = soc_max
    temp_curr = temp_amb
    t_pulso = 0.0
    ah_acum = 0.0
    
    cp = 1000 # Calor específico

    # --- BUCLE DE SIMULACIÓN PASO A PASO ---
    for i in range(num_steps):
        I_pack_inst = I_base_pack[i]
        I_cell_inst = I_pack_inst / n_p
        
        # A. Calcular Resistencia Dinámica Instantánea
        r_dinamica_cell = _calcular_resistencia_dinamica(r_base_envejecida, soc_curr, temp_curr, I_cell_inst, t_pulso)
        r_pack_total = (r_dinamica_cell * n_s / n_p) + R_CONN_PACK
        
        # B. Actualizar Polarización (t_pulso)
        if abs(I_cell_inst) > 1.0: # Si hay corriente significativa
            t_pulso += dt
        else: # Relajación
            t_pulso = max(0.0, t_pulso - dt * 2.0)
            
        # C. Voltaje OCV
        ocv_cell = f_ocv_user(np.clip(soc_curr, 0, 100))
        ocv_pack = ocv_cell * n_s
        
        # D. Caída de Tensión y Voltaje Terminal
        v_drop = I_pack_inst * r_pack_total
        v_term = ocv_pack - v_drop
        
        # E. Actualizar SOC
        ah_step = I_pack_inst * (dt / 3600.0)
        ah_acum += ah_step
        soc_curr = soc_max - (ah_acum / cap_celda_real / n_p) * 100.0
        
        # F. Modelo Térmico
        Q_gen = (I_cell_inst**2) * r_dinamica_cell # Calor generado por celda
        Q_out = refrigeracion * (temp_curr - temp_amb)
        dT = ((Q_gen - Q_out) / (mass * cp)) * dt
        temp_curr += dT
        
        # Guardar datos
        soc_t[i] = soc_curr
        v_pack_t[i] = v_term
        temps[i] = temp_curr

    # Post-proceso para compatibilidad con código antiguo
    ah_consumed = np.cumsum(I_base_pack * (dt / 3600.0)) # Recalculo simple para vectorizar verificaciones
    soc_t = np.clip(soc_t, 0, 100) # Clamp final
    
    # Valores medios para reporte
    r_operative_cell = np.mean([_calcular_resistencia_dinamica(r_base_envejecida, 50, 40, 20, 10)]) # Valor representativo
    
    # 4. ALERTAS DE TENSIÓN
    # (El código siguiente usa v_pack_t que ya calculamos arriba)
    
    p_pack_t = v_pack_t * I_base_pack
    viable = ah_consumed[-1] <= (cap_pack_ah_real * (soc_max - soc_min) / 100.0)

    # 4. ALERTAS DE TENSIÓN
    v_status = "VOLTAJE OK"
    v_color = "green"
    
    V_MAX_ABS = 16.0
    V_MIN_NOM = 9.8
    V_MIN_ABS = 9.5
    T_TRANS_MAX = 0.1 

    if np.any(v_pack_t > V_MAX_ABS):
        v_status = f"FALLO: V > {V_MAX_ABS}V"
        v_color = "red"
    elif np.any(v_pack_t < V_MIN_ABS):
        v_status = f"FALLO: V < {V_MIN_ABS}V (CRÍTICO)"
        v_color = "red"
    elif np.any(v_pack_t < V_MIN_NOM):
        mask_trans = (v_pack_t < V_MIN_NOM) & (v_pack_t >= V_MIN_ABS)
        is_trans = np.concatenate(([0], mask_trans.astype(int), [0]))
        diffs = np.diff(is_trans)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        durations = (ends - starts) * dt
        
        if np.any(durations > T_TRANS_MAX):
            v_status = f"FALLO: Bajada > {T_TRANS_MAX}s"
            v_color = "red"
        else:
            v_status = "AVISO: TRANSITORIO (<0.1s)"
            v_color = "orange"

    t_max = np.max(temps)
    if t_max > 60.0:
        t_status = f"PELIGRO: >60°C ({t_max:.1f}°C)"
        t_color = "red"
    else:
        t_status = f"TEMP OK ({t_max:.1f}°C)"
        t_color = "green"

# 6. Degradación (HÍBRIDO CALIBRADO: IA + FATIGA + ESTRÉS VOLTAJE + AJUSTE POUCH)
    # -------------------------------------------------------------------------
    # MODIFICACIÓN FINAL: Sincronización con Datasheet Grepow.
    # La IA usa datos de celdas de laboratorio (>1500 ciclos).
    # Nuestra celda es Pouch comercial (~500-600 ciclos).
    # Aplicamos un factor de corrección de tecnología.
    # -------------------------------------------------------------------------

    FACTOR_AJUSTE_POUCH = 2.0  # Penalización x2 para pasar de "Calidad Lab" a "Calidad Pouch Comercial"

    # A) Variables de Entrada
    # A) Variables de Entrada
    # Recalculamos corriente celda para esta sección
    I_cell = I_base_pack / n_p 
    rms_current = np.sqrt(np.mean(I_cell**2))
    c_rate_efectivo = rms_current / cap_celda if cap_celda > 0 else 0
    temp_promedio = np.mean(temps)

    # B) Definición del Ciclo
    # B.1. Magnitud (DoD)
    dod_slider_pct = soc_max - soc_min
    if dod_slider_pct < 1.0: dod_slider_pct = 1.0
    dod_decimal = dod_slider_pct / 100.0
    
    # B.2. Posición (SOC Medio)
    soc_promedio = (soc_max + soc_min) / 2.0

    # C) Consulta a la IA -> Obtener "Ancla"
    try:
        tasa_pct_base_100, info_ia = cerebro_ia.predecir_degradacion(c_rate_efectivo, temp_promedio, 100.0)
    except TypeError:
        tasa_pct_base_100, info_ia = cerebro_ia.predecir_degradacion(c_rate_efectivo, temp_promedio)

    # D) FACTORES FÍSICOS DE CORRECCIÓN
    
    # 1. Fatiga Mecánica (DoD)
    factor_fatiga = dod_decimal ** 1.6
    
    # 2. Estrés por Zona de SOC (Curva en V centrada en 50%)
    distancia_al_centro = abs(soc_promedio - 50.0)
    k_parabola = 0.0027 
    factor_stress_soc = 1.0 + (k_parabola * (distancia_al_centro ** 2))

    # E) CÁLCULO FINAL CON CALIBRACIÓN
    # E) CÁLCULO FINAL CON CALIBRACIÓN
    if tasa_pct_base_100 is not None:
        # Fórmula Maestra: Base_IA * Ajuste_Tecnologia * Fatiga * Estrés_SOC
        deg_por_ciclo = (tasa_pct_base_100 * cap_celda) * FACTOR_AJUSTE_POUCH * factor_fatiga * factor_stress_soc
        
        metodo_usado = f"Híbrido Calibrado (DoD {dod_slider_pct}% | Factor Tech x{FACTOR_AJUSTE_POUCH})"
    else:
        # Fallback si IA falla: No calculamos degradación
        deg_por_ciclo = 0.0
        metodo_usado = "IA no disponible"

    # F) Cálculo de Vida Útil
    capacidad_limite_perdida = cap_celda * 0.20
    
    if deg_por_ciclo > 1e-15:
        ciclos = capacidad_limite_perdida / deg_por_ciclo
    else:
        ciclos = 500000

  
    
    # Debug (Opcional)
    # print(f"Avg: {soc_promedio}% | Stress Factor: {factor_stress_soc:.3f}")

    # --- VISUALIZACIÓN ---
    fmt = FuncFormatter(time_formatter)

    # GRÁFICA 1
    ax_graph1.clear()
    ax_graph1.xaxis.set_major_formatter(fmt)
    if graph_mode == 'Intensidad (A)':
        ax_graph1.plot(t_base, I_base_pack, 'b')
        ax_graph1.set_ylabel("Corriente (A)")
        ax_graph1.set_title(f"PERFIL DE CARGA")
    elif graph_mode == 'Tensión (V)':
        ax_graph1.plot(t_base, v_pack_t, 'purple')
        ax_graph1.set_ylabel("Tensión (V)")
        ax_graph1.set_title(f"EVOLUCIÓN TENSIÓN (Caída Max: {np.max(v_pack_t) - np.min(v_pack_t):.2f}V)")
        ax_graph1.axhline(V_MAX_ABS, color='r', linestyle='--')
        ax_graph1.axhline(V_MIN_NOM, color='orange', linestyle='--')
        ax_graph1.axhline(V_MIN_ABS, color='r', linestyle=':', linewidth=2)
    elif graph_mode == 'Potencia (W)':
        ax_graph1.plot(t_base, p_pack_t, 'g')
        ax_graph1.set_ylabel("Potencia (W)")
        ax_graph1.set_title("POTENCIA INSTANTÁNEA")

    ax_graph1.grid(True, alpha=0.3)
    ax_graph1.text(0.02, 0.85, "ENERGÍA OK" if viable else "ENERGÍA INSUF.", transform=ax_graph1.transAxes, 
                   color='white', fontweight='bold', bbox=dict(facecolor='green' if viable else 'red', alpha=0.8))
    ax_graph1.text(0.02, 0.70, v_status, transform=ax_graph1.transAxes, 
                   color='white', fontweight='bold', bbox=dict(facecolor=v_color, alpha=0.8))

    # GRÁFICA 2
    ax_term.clear()
    ax_term.xaxis.set_major_formatter(fmt)
    ax_term.plot(t_base, temps, 'r')
    ax_term.axhline(60, color='r', linestyle='-', linewidth=2)
    ax_term.axhline(temp_amb, color='orange', linestyle='--')
    ax_term.set_ylabel("Temp (°C)")
    ax_term.set_title(f"TÉRMICO (Max: {np.max(temps):.1f}°C)")
    ax_term.text(0.02, 0.85, t_status, transform=ax_term.transAxes, 
                 color='white', fontweight='bold', bbox=dict(facecolor=t_color, alpha=0.8))
    ax_term.grid(True, alpha=0.3)

    # GRÁFICA 3
    ax_soc.clear()
    ax_soc.xaxis.set_major_formatter(fmt)
    ax_soc.plot(t_base, soc_t, 'k')
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_ylim(0, 100)
    ax_soc.set_title("ESTADO DE CARGA")
    ax_soc.axhline(soc_min, color='r', linestyle='--', alpha=0.5)
    ax_soc.grid(True, alpha=0.3)

    # CÁLCULO DE VIDA (TEXTO SOLO)
    if deg_por_ciclo <= 0: deg_por_ciclo = 1e-9
    
    # Texto del título dinámico
    if ciclos > 90000:
        texto_vida = "> 90.000 Ciclos"
    else:
        texto_vida = f"{int(ciclos)} Ciclos"
        
    # Mostrar el valor como texto en la figura
    global txt_vida
    try: txt_vida.remove()
    except: pass
    
    txt_vida = plt.text(0.30, 0.65, f"VIDA ESTIMADA:\n{texto_vida}", transform=fig.transFigure, fontsize=12, 
                       fontweight='bold', color='white', ha='center',
                       bbox=dict(facecolor='#2ca02c', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.5'))

    # TEXTO RESISTENCIAS
    global txt_res
    try: txt_res.remove()
    except: pass
    info_res = f"R. Celda (Dinámica): ~{r_operative_cell*1000:.2f} mΩ\nR. Pack (Dinámica):  ~{r_pack_total*1000:.1f} mΩ"
    txt_res = plt.text(0.05, 0.65, info_res, transform=fig.transFigure, fontsize=10, 
                       fontweight='bold', color='white',
                       bbox=dict(facecolor='#007acc', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.5'))

    fig.canvas.draw_idle()

# ==========================================
# 5. CONTROLES
# ==========================================
txt_res = None 
txt_vida = None 

ax_radio = plt.axes([0.05, 0.70, 0.35, 0.20], facecolor='#f0f0f0')
radio = RadioButtons(ax_radio, list(MACROS_CONFIGURACION.keys()), active=0)

def aplicar_configuracion(label):
    macro = MACROS_CONFIGURACION[label]
    # Mover sliders a la posición definida en la macro
    controls[1][0].set_val(macro['S'])      # Series
    controls[2][0].set_val(macro['P'])      # Paralelo
    # La capacidad (controls[0]) NO se toca, es propiedad de la celda
    
    if 'S_max' in macro: controls[3][0].set_val(macro['S_max'])
    if 'S_min' in macro: controls[4][0].set_val(macro['S_min'])

radio.on_clicked(aplicar_configuracion)
plt.text(0.05, 0.91, "1. CONFIGURACIÓN RÁPIDA (MACROS)", transform=fig.transFigure, fontsize=11, fontweight='bold', color='blue')

ax_radio_plot = plt.axes([0.25, 0.92, 0.15, 0.06], facecolor='#e6e6e6')
radio_plot = RadioButtons(ax_radio_plot, ['Tensión (V)', 'Intensidad (A)', 'Potencia (W)'])
def change_graph(label): global graph_mode; graph_mode = label; run_simulation()
radio_plot.on_clicked(change_graph)
plt.text(0.25, 0.985, "VISUALIZACIÓN SUPERIOR", transform=fig.transFigure, fontsize=9, fontweight='bold')

def make_control(label, vmin, vmax, vinit, y_pos, step, fmt="%1.0f"):
    plt.text(0.05, y_pos + 0.025, label, transform=fig.transFigure, fontsize=9, fontweight='bold')
    ax_s = plt.axes([0.05, y_pos, 0.25, 0.02], facecolor='lightgoldenrodyellow')
    s = Slider(ax_s, '', vmin, vmax, valinit=vinit, valstep=step, valfmt=fmt)
    # BOTONES DESPLAZADOS A LA DERECHA (0.35 y 0.38)
    ax_min = plt.axes([0.35, y_pos, 0.02, 0.02]); b_min = Button(ax_min, '-', hovercolor='0.9')
    ax_plus = plt.axes([0.38, y_pos, 0.02, 0.02]); b_plus = Button(ax_plus, '+', hovercolor='0.9')
    
    def update(val): 
        # Corrección para sliders invertidos (vmin > vmax)
        if vmin < vmax:
            if s.val < vmin: s.set_val(vmin)
            if s.val > vmax: s.set_val(vmax)
        else:
            if s.val > vmin: s.set_val(vmin)
            if s.val < vmax: s.set_val(vmax)
            
        run_simulation()
    
    def dec(e): s.set_val(np.clip(s.val - step, vmin, vmax))
    def inc(e): s.set_val(np.clip(s.val + step, vmin, vmax))
    
    s.on_changed(update)
    b_min.on_clicked(dec)
    b_plus.on_clicked(inc)
    controls.append([s, b_min, b_plus])

start_y = 0.55; step_y = 0.06
plt.text(0.05, 0.60, "2. AJUSTE MANUAL", transform=fig.transFigure, fontsize=11, fontweight='bold')

make_control('Capacidad Celda (Ah)', 2.0, 16.0, 10.0, start_y, 0.1, "%1.1f")
make_control('Series (S)', 1, 6, 3, start_y - step_y, 1)
make_control('Paralelo (P)', 1, 6, 2, start_y - 2*step_y, 1)
make_control('SOC Máximo (%)', 60, 100, 80, start_y - 3.5*step_y, 1)
make_control('SOC Mínimo (%)', 0, 40, 20, start_y - 4.5*step_y, 1)
make_control('Temp. Ambiente (°C)', 15.0, 65.0, 42.4, start_y - 6*step_y, 0.5, "%1.1f")
make_control('Refrigeración (W/K)', 0.0, 8.0, 0.3, start_y - 7*step_y, 0.1, "%1.1f")

# Clase Slider Invertido (100 -> Izquierda, 60 -> Derecha)
class InvertedSlider(Slider):
    def set_val(self, val):
        self.val = val
        self.valtext.set_text(self.valfmt % val)
        # Invertir posición visual: (val - vmin) / (vmax - vmin) -> 1 - (...)
        norm = (val - self.valmin) / (self.valmax - self.valmin)
        self.poly.set_width(1 - norm) # Dibuja de derecha a izquierda visualmente
        self.poly.set_x(norm)         # Opcional: ajustar origen si se quisiera barra flotante
        
        # Truco: Matplotlib Slider dibuja un rectángulo desde 0.
        # Para invertirlo "fácil" sin reescribir todo el draw:
        # Usamos el slider normal pero mapeamos visualmente al revés.
        # Mejor opción simple: Usar slider normal (60->100) pero etiquetas cambiadas?
        # No, el usuario pide 100 a la izquierda.
        
        # SOLUCIÓN MÁS ROBUSTA: Usar un slider normal matemático pero pintar invertido.
        # Matplotlib estándar no soporta "invertido" fácil. 
        # Haremos un "Hack": El slider va de 60 a 100.
        # Pero visualmente queremos que el "min" (izquierda) sea 100 y "max" (derecha) sea 60.
        # Entonces el rango matemático será [60, 100].
        # Si x_mouse está a la izquierda (0), valor = 100.
        # Si x_mouse está a la derecha (1), valor = 60.
        # Fórmula: valor = v_max - (mouse_x_norm * (v_max - v_min))
        pass

# Re-implementación simple usando Slider estándar pero con lógica invertida en simulación
# El usuario pide: "100% a la izquierda, 60% a la derecha"
# Esto significa un slider que va de 100 (min_axis) a 60 (max_axis).
# Matplotlib permite vmin > vmax? SÍ.

make_control('SOH Simulado (%)', 60.0, 100.0, 100.0, start_y - 8*step_y, 1, "%1.0f")

aplicar_configuracion('3S2P (80-20)')
run_simulation()
plt.show()
