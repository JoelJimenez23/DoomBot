---

# 🧠 DoomBot – RL + Computer Vision en VizDoom

Bot autónomo para **VizDoom** basado en **Reinforcement Learning (PPO)** y **visión por computadora**, diseñado para aprender a jugar Doom usando únicamente entradas visuales (cumpliendo restricciones académicas).

---

## 🚀 Features principales

* ✅ Entrenamiento con **PPO (Stable-Baselines3)**
* ✅ Entorno custom tipo Gym (`DoomTaskEnv`)
* ✅ Arquitectura modular:

  * `core`: interacción con VizDoom
  * `tasks`: definición de comportamientos
  * `features`: extracción de información visual
  * `rewards`: diseño de recompensas
* ✅ Soporte multi-escenario (familias de mapas)
* ✅ Evaluación automática y manual
* ✅ Integración inicial con **YOLO (detección visual)**
* ⚠️ Sin uso de estado interno del juego en inferencia (según restricciones)

---

## 📂 Estructura del proyecto

```
DoomBot/
│
├── doom_env/                # Entorno principal
│   ├── core/               # Interacción con VizDoom
│   │   ├── vizdoom_core.py
│   │   ├── state.py
│   │   └── action_space.py
│   │
│   ├── tasks/              # Definición de tareas RL
│   │   ├── base_task.py
│   │   └── aim_shoot.py
│   │
│   ├── features/           # Extracción de features
│   │   ├── enemies.py
│   │   ├── pickups.py
│   │   └── detectors.py
│   │
│   ├── rewards/            # Funciones de recompensa
│   │   ├── aim_reward.py
│   │   └── pickups_reward.py
│   │
│   └── env.py              # Wrapper tipo Gym
│
├── eval/                  # Evaluación del agente
│   ├── bot_evaluation.py
│   ├── debug_bot_evaluation.py
│   └── manual_evaluation.py
│
├── models/                # Modelos entrenados
│   ├── ppo_doom_center.zip
│   ├── ppo_doom_deadly_corridor_multitask.zip
│   ├── ...
│
├── scenarios/             # Mapas de VizDoom (.wad + .cfg)
│
├── _vizdoom.ini
└── README.md
```

---

## 🧠 Arquitectura

El flujo del sistema es:

```
Imagen (VizDoom)
     ↓
Feature Extraction (YOLO / heurísticas)
     ↓
Task (define objetivo)
     ↓
Reward Function
     ↓
PPO Policy
     ↓
Acciones (VizDoom)
```

---

## 🎮 Entorno

El entorno principal es:

```python
DoomTaskEnv(cfg, task)
```

Donde:

* `cfg`: configuración del entorno (`EnvConfig`)
* `task`: define el comportamiento (ej: `TaskAimShoot`)

---

## ⚙️ Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

Dependencias principales:

* `vizdoom`
* `stable-baselines3`
* `torch`
* `opencv-python`
* `numpy`

---

## 🏋️ Entrenamiento

Ejemplo básico:

```python
from stable_baselines3 import PPO
from doom_env import DoomTaskEnv
from doom_env.tasks.aim_shoot import TaskAimShoot

env = DoomTaskEnv(cfg, TaskAimShoot())

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=800_000)

model.save("models/my_model")
```

---

## 📊 Evaluación

### Automática

```bash
python eval/bot_evaluation.py
```

### Manual (control humano)

```bash
python eval/manual_evaluation.py
```

---

## 🧪 Escenarios

Se usan escenarios de VizDoom:

* `basic`
* `defend_the_center`
* `defend_the_line`
* `deadly_corridor`
* `health_gathering`

Configurados vía `.cfg` y `.wad`.

---

## 🧩 Tasks

Las tareas definen el comportamiento del agente:

Ejemplo:

* `TaskAimShoot`

  * Apuntar al enemigo
  * Disparar eficientemente
  * Maximizar precisión

---

## 🎯 Rewards

Diseño modular de recompensas:

* `aim_reward`: precisión de disparo
* `pickups_reward`: recolección de recursos

---

## 👁️ Visión por computadora

Actualmente:

* Features basadas en heurísticas (bounding boxes, dx/dy)
* Integración en progreso con YOLO (`best.pt`)

⚠️ Restricción importante:

> No se permite usar variables internas del juego en inferencia.

---

## 📈 Modelos entrenados

Incluye varios checkpoints:

* `ppo_doom_center`
* `ppo_doom_deadly_corridor_multitask`
* `ppo_multimap_yolo_ft`
* etc.

---

## 🚧 Estado actual del proyecto

### ✔️ Implementado

* Entorno RL funcional
* Entrenamiento PPO
* Evaluación
* Multi-escenario
* Modularidad completa

### ⚠️ En progreso

* Integración completa de YOLO
* Generalización entre mapas
* Exploración (evitar quedarse quieto)
* Mejor uso de munición

---

## 🧠 Ideas futuras

* 🔥 Aceleradores hardware (FPGA / SoC para visión)
* 🧠 Multi-task learning real (familias A/B/C)
* 🎯 Predicción de movimiento de enemigos
* 🌍 Entrenamiento generalizado multi-mapa
* ⚡ Optimización del pipeline de inferencia

---

## 👤 Autor

Joel Jimenez
Estudiante de Ciencias de la Computación – UTEC

Intereses:

* Sistemas operativos
* Arquitectura de computadoras
* IA eficiente / aceleradores

---
