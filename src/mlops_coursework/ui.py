import json
import os
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.getenv("API_URL", "[http://127.0.0.1:8000](http://127.0.0.1:8000)")

st.set_page_config(page_title="AI4I Predictive Maintenance", layout="centered")
st.title("AI4I 2020 Predictive Maintenance")
st.caption(f"Inference endpoint: `{API_URL}/predict`")

with st.form("sensor_form"):
    machine_type = st.selectbox("Тип оборудования (Machine Type)", ["L", "M", "H"], index=1)
    air_temp_k = st.number_input("Температура воздуха [K]", 250.0, 350.0, 298.1)
    process_temp_k = st.number_input("Температура процесса [K]", 250.0, 400.0, 308.6)
    rotational_speed_rpm = st.number_input("Скорость вращения [rpm]", 1, 5000, 1551)
    torque_nm = st.number_input("Крутящий момент [Nm]", 0.0, 150.0, 42.8)
    tool_wear_min = st.number_input("Износ инструмента [min]", 0, 400, 108)
    submitted = st.form_submit_button("Рассчитать риск отказа")

if submitted:
    payload = json.dumps(
        {
            "machine_type": machine_type,
            "air_temp_k": air_temp_k,
            "process_temp_k": process_temp_k,
            "rotational_speed_rpm": int(rotational_speed_rpm),
            "torque_nm": torque_nm,
            "tool_wear_min": int(tool_wear_min),
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{API_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        status_label = "ОТКАЗ (1)" if result["failure_prediction"] == 1 else "НОРМА (0)"
        st.success(f"Прогноз: {status_label}")
        st.metric("Вероятность отказа", f"{result['failure_probability'] * 100:.2f}%")
        st.write(
            f"Уровень риска: **{result['risk_level']}** | "
            f"Версия модели: `{result['model_version']}`"
        )
    except urllib.error.HTTPError as err:
        st.error(f"Ошибка валидации API ({err.code}): {err.read().decode('utf-8')}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        st.error(f"Не удалось подключиться к API: {exc}")
