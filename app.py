from __future__ import annotations

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from tomography import (
    LayeredVelocityModel,
    VelocityModel,
    create_toy_scenario,
    gradient_descent_update,
    hodograph_samples,
    misfit,
    project_to_bounds,
    shoot_head_wave,
    simple_layered_start,
    gradient_velocity_field,
    sirt_update,
    predict_times_linear,
    travel_time_matrix,
)


st.set_page_config(page_title="Сейсмотомография — интерактивное объяснение", layout="wide")
st.title("Сейсмотомография: от годографа до градиентного обновления")
st.markdown(
    """
    Этот интерактив помогает наглядно объяснить другу этапы
    **продольной сейсмотомографии**: аналитический годограф рефрагированных волн,
    прямолинейные лучи для инверсии, вычисление градиента и один шаг обновления
    модели. Работаем в 2D разрезе, а для простоты используем прямолучевой подход.
    """
)

DEFAULT_KIND = "gradient"
DEFAULT_SEED = 7
DEFAULT_SAMPLES = 150


def _build_reference_field(scenario_obj, layer_velocities):
    meta = scenario_obj.metadata or {}
    start_type = meta.get("start_model_type", "layered")
    if start_type == "gradient":
        v0 = float(st.session_state.get("grad_v0", meta.get("start_gradient", {}).get("v0", 1800.0)))
        gz = float(st.session_state.get("grad_gz", meta.get("start_gradient", {}).get("gz", 0.6)))
        gx = float(st.session_state.get("grad_gx", meta.get("start_gradient", {}).get("gx", 0.0)))
        return gradient_velocity_field(scenario_obj.true_model.x, scenario_obj.true_model.z, v0=v0, gz=gz, gx=gx)
    interfaces = tuple(meta.get("start_interface_depths", ()))
    return simple_layered_start(
        scenario_obj.true_model.x,
        scenario_obj.true_model.z,
        velocities=tuple(layer_velocities),
        interface_depths=interfaces,
    )


def _reset_state(scenario_obj, layer_velocities, n_samples):
    reference_field = _build_reference_field(scenario_obj, layer_velocities)
    scenario_obj.starting_model = VelocityModel(
        scenario_obj.true_model.x,
        scenario_obj.true_model.z,
        reference_field.copy(),
    )
    st.session_state.reference_velocity = reference_field
    st.session_state.velocity = reference_field.copy()
    st.session_state.predicted = travel_time_matrix(
        scenario_obj.starting_model,
        scenario_obj.geometry,
        n_samples=n_samples,
    )
    st.session_state.gradient = np.zeros_like(reference_field)
    st.session_state.iteration = 0
    st.session_state.misfit_history = []
    st.session_state.ray_sampling = n_samples


if "scenario_kind" not in st.session_state:
    initial_scenario = create_toy_scenario(kind=DEFAULT_KIND, random_state=DEFAULT_SEED)
    base_layers = list(initial_scenario.metadata.get("start_layer_velocities", (2000.0,)))
    st.session_state.scenario_kind = DEFAULT_KIND
    st.session_state.scenario_seed = DEFAULT_SEED
    st.session_state.scenario = initial_scenario
    st.session_state.start_layer_velocities = base_layers
    _reset_state(initial_scenario, base_layers, DEFAULT_SAMPLES)

scenario = st.session_state.scenario

sidebar = st.sidebar
sidebar.header("Сценарий модели")
kind_labels = {
    "gradient": "Градиентная",
    "geologic": "Геологическая (плавающие границы)",
    "simple": "Простая горизонтальная",
}
current_kind_label = kind_labels.get(st.session_state.scenario_kind, kind_labels[DEFAULT_KIND])
selected_label = sidebar.selectbox(
    "Выбор сценария",
    options=list(kind_labels.values()),
    index=list(kind_labels.values()).index(current_kind_label),
)
inverse_kind = {v: k for k, v in kind_labels.items()}
seed_input = sidebar.number_input(
    "Сид генерации",
    value=int(st.session_state.scenario_seed),
    step=1,
)
regenerate = sidebar.button("Перегенерировать модель")

geom_label = sidebar.selectbox(
    "Геометрия съёмки",
    options=["поверхностная", "скважинная (crosshole)"],
    index=1,
)
use_linear = sidebar.checkbox("Линеаризованный форвард (SIRT)", value=True,
    help="Если включено, наблюдения и прогноз считаются через T=G·s, как в SIRT-обновлении.")

selected_kind = inverse_kind[selected_label]
seed_int = int(seed_input)
if (
    regenerate
    or selected_kind != st.session_state.scenario_kind
    or seed_int != st.session_state.scenario_seed
):
    new_scenario = create_toy_scenario(kind=selected_kind, random_state=seed_int)
    meta = new_scenario.metadata or {}
    new_layers = list(meta.get("start_layer_velocities", ()))
    if meta.get("start_model_type") == "gradient":
        st.session_state["grad_v0"] = float(meta.get("start_gradient", {}).get("v0", 1800.0))
        st.session_state["grad_gz"] = float(meta.get("start_gradient", {}).get("gz", 0.6))
        st.session_state["grad_gx"] = float(meta.get("start_gradient", {}).get("gx", 0.0))
    st.session_state.scenario_kind = selected_kind
    st.session_state.scenario_seed = seed_int
    st.session_state.scenario = new_scenario
    st.session_state.start_layer_velocities = new_layers
    for key in list(st.session_state.keys()):
        if key.startswith("layer_vel_"):
            del st.session_state[key]
    for idx, val in enumerate(new_layers):
        st.session_state[f"layer_vel_{idx}"] = float(val)
    st.session_state.pop("shoot_guess_along", None)
    st.session_state.pop("shoot_target_offset", None)
    _reset_state(new_scenario, new_layers, st.session_state.get("ray_sampling", DEFAULT_SAMPLES))
    scenario = new_scenario

scenario = st.session_state.scenario

# Apply geometry choice and recompute observations consistently
from tomography.geometry import AcquisitionGeometry
def _apply_geometry_choice():
    global scenario
    if geom_label.startswith("поверхностная"):
        geom = AcquisitionGeometry.surface_spread(
            n_sources=14, n_receivers=14, spread_length=scenario.true_model.x.max()
        )
    else:
        geom = AcquisitionGeometry.crosshole(
            n_sources=18,
            n_receivers=18,
            x_left=scenario.true_model.x.min(),
            x_right=scenario.true_model.x.max(),
            z_top=scenario.true_model.z.min(),
            z_bottom=scenario.true_model.z.max(),
        )
    scenario.geometry = geom
    # Recompute observations using selected forward
    if use_linear:
        obs = predict_times_linear(scenario.true_model, geom, n_samples=st.session_state.get("ray_sampling", DEFAULT_SAMPLES))
    else:
        obs = travel_time_matrix(scenario.true_model, geom, n_samples=st.session_state.get("ray_sampling", DEFAULT_SAMPLES))
    # Reset predictions for current velocity too
    if use_linear:
        pred = predict_times_linear(
            VelocityModel(scenario.true_model.x, scenario.true_model.z, st.session_state.velocity),
            geom,
            n_samples=st.session_state.get("ray_sampling", DEFAULT_SAMPLES),
        )
    else:
        pred = travel_time_matrix(
            VelocityModel(scenario.true_model.x, scenario.true_model.z, st.session_state.velocity),
            geom,
            n_samples=st.session_state.get("ray_sampling", DEFAULT_SAMPLES),
        )
    # Stash in session
    st.session_state.predicted = pred
    # Patch scenario's observations used by the update
    scenario = st.session_state.scenario
    scenario.observed_travel_times = obs

_apply_geometry_choice()

sidebar.subheader("Стартовая модель")
meta = scenario.metadata or {}
if meta.get("start_model_type") == "gradient":
    st.session_state.setdefault("grad_v0", float(meta.get("start_gradient", {}).get("v0", 1800.0)))
    st.session_state.setdefault("grad_gz", float(meta.get("start_gradient", {}).get("gz", 0.6)))
    st.session_state.setdefault("grad_gx", float(meta.get("start_gradient", {}).get("gx", 0.0)))
    v0_val = sidebar.number_input("v₀ на поверхности (м/с)", value=float(st.session_state["grad_v0"]))
    gz_val = sidebar.number_input("Вертикальный градиент g_z (м/с на м)", value=float(st.session_state["grad_gz"]))
    gx_val = sidebar.number_input("Боковой градиент g_x (м/с на м)", value=float(st.session_state["grad_gx"]))
    if sidebar.button("Применить стартовую модель"):
        st.session_state["grad_v0"] = float(v0_val)
        st.session_state["grad_gz"] = float(gz_val)
        st.session_state["grad_gx"] = float(gx_val)
        st.session_state.pop("shoot_guess_along", None)
        st.session_state.pop("shoot_target_offset", None)
        _reset_state(scenario, [], st.session_state.get("ray_sampling", DEFAULT_SAMPLES))
else:
    start_layers = st.session_state.start_layer_velocities
    layer_controls = []
    for idx, base in enumerate(start_layers):
        key = f"layer_vel_{idx}"
        st.session_state.setdefault(key, float(base))
        layer_controls.append(
            sidebar.slider(
                f"Скорость слоя {idx + 1} (м/с)",
                1000.0,
                4500.0,
                float(st.session_state[key]),
                step=10.0,
                key=key,
            )
        )
    if sidebar.button("Применить стартовую модель"):
        updated_layers = [float(st.session_state[f"layer_vel_{idx}"]) for idx in range(len(layer_controls))]
        st.session_state.start_layer_velocities = updated_layers
        st.session_state.pop("shoot_guess_along", None)
        st.session_state.pop("shoot_target_offset", None)
        _reset_state(scenario, updated_layers, st.session_state.get("ray_sampling", DEFAULT_SAMPLES))

sidebar.header("Параметры инверсии")
step_size = sidebar.slider(
    "Максимальная поправка скорости за шаг (м/с)", 1.0, 200.0, 40.0
)
n_samples = sidebar.slider(
    "Дискретизация луча (точек)",
    min_value=30,
    max_value=400,
    value=st.session_state.get("ray_sampling", 120),
    step=10,
)
if n_samples != st.session_state.get("ray_sampling", DEFAULT_SAMPLES):
    st.session_state.ray_sampling = n_samples
    st.session_state.predicted = travel_time_matrix(
        VelocityModel(
            scenario.true_model.x,
            scenario.true_model.z,
            st.session_state.velocity,
        ),
        scenario.geometry,
        n_samples=n_samples,
    )
reg_lambda = sidebar.slider(
    "λ Тихонова (0 — без регуляризации)", 0.0, 0.01, 0.002, step=0.0005, format="%.4f"
)
clip_min = sidebar.number_input("Мин. скорость (м/с)", value=1500.0, min_value=500.0)
clip_max = sidebar.number_input("Макс. скорость (м/с)", value=4000.0, min_value=clip_min + 10.0)

with sidebar.expander("Что даёт Тихоновская регуляризация?"):
    st.markdown(
        """
        Тихоновскую (L2) регуляризацию можно представить как «резиновый шнур»,
        подтягивающий модель к *референсу*. Здесь референсом служит стартовая
        модель. Чем больше λ, тем сильнее штраф за отклонение от неё и тем
        плавнее становится итоговая скорость.
        """
    )

with sidebar.expander("Параметры годографа и лучей"):
    v1 = sidebar.number_input("Скорость верхнего слоя v₁ (м/с)", value=2000.0, min_value=500.0)
    depth = sidebar.slider("Глубина границы (м)", 100.0, 1500.0, 600.0, step=50.0)
    v2 = sidebar.number_input("Скорость ниже границы v₂ (м/с)", value=3500.0, min_value=v1 + 10.0)
    max_offset = sidebar.slider(
        "Максимальное расстояние (м)", 500.0, 6000.0, 3000.0, step=100.0
    )
    single_offset = sidebar.slider(
        "Смещение приёмника для луча (м)", 0.0, max_offset, 2500.0, step=100.0
    )

col_reset, col_step = sidebar.columns(2)
if col_reset.button("Сбросить модель"):
    st.session_state.velocity = scenario.starting_model.velocity.copy()
    st.session_state.predicted = travel_time_matrix(
        scenario.starting_model, scenario.geometry, n_samples=n_samples
    )
    st.session_state.gradient = np.zeros_like(st.session_state.velocity)
    st.session_state.iteration = 0
    st.session_state.misfit_history = []

if col_step.button("Сделать шаг"):
    working_model = VelocityModel(
        scenario.starting_model.x, scenario.starting_model.z, st.session_state.velocity
    )
    # Use SIRT-like update in slowness domain with optional smoothing for stability
    updated, predicted, gradient = sirt_update(
        model=working_model,
        geometry=scenario.geometry,
        observed=scenario.observed_travel_times,
        step_size=step_size,
        n_samples=n_samples,
        tikhonov_lambda=reg_lambda,
        smooth_weight=sidebar.slider("Сглаживание (лапл.)", 0.0, 0.01, 0.002, step=0.0005),
    )
    updated = project_to_bounds(updated, clip_min, clip_max)
    updated_model = VelocityModel(
        scenario.starting_model.x,
        scenario.starting_model.z,
        updated,
    )
    # Keep predicted consistent with the linearised forward used by SIRT
    if use_linear:
        new_predicted = predict_times_linear(updated_model, scenario.geometry, n_samples=n_samples)
    else:
        new_predicted = travel_time_matrix(updated_model, scenario.geometry, n_samples=n_samples)
    st.session_state.velocity = updated
    st.session_state.predicted = new_predicted
    st.session_state.gradient = gradient
    st.session_state.iteration += 1

current_model = VelocityModel(
    scenario.true_model.x, scenario.true_model.z, st.session_state.velocity
)

misfit_value, residual = misfit(
    st.session_state.predicted, scenario.observed_travel_times
)
if len(st.session_state.misfit_history) <= st.session_state.iteration:
    st.session_state.misfit_history.append(misfit_value)
else:
    st.session_state.misfit_history[st.session_state.iteration] = misfit_value
st.markdown(
    f"**Текущий шаг:** {st.session_state.iteration} — "
    f"несовпадение годографов (½‖r‖²) = `{misfit_value:.3e}`"
)

layered_model = LayeredVelocityModel(
    velocities=np.array([v1, v2]),
    thicknesses=np.array([depth, np.inf]),
)
offsets = np.linspace(0.0, max_offset, 200)
hodograph = hodograph_samples(layered_model, offsets)
theta_c, intercept_time, crossover_offset = layered_model.head_wave_parameters(0)
horizontal_leg = depth * np.tan(theta_c)


def _plot_velocity(model: VelocityModel, title: str, overlay_interfaces: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    mesh = ax.pcolormesh(
        model.x,
        model.z,
        model.velocity,
        shading="auto",
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.invert_yaxis()
    if overlay_interfaces:
        meta = scenario.metadata or {}
        x_ifc = meta.get("interfaces_x")
        z_ifcs = meta.get("interfaces_z")
        if x_ifc is not None and z_ifcs is not None:
            for zc in z_ifcs:
                ax.plot(x_ifc, zc, color="white", lw=2.5, alpha=0.9)
                ax.plot(x_ifc, zc, color="black", lw=1.0, alpha=0.8)
    fig.colorbar(mesh, ax=ax, label="м/с")
    fig.tight_layout()
    return fig


def _plot_gradient(gradient: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    mesh = ax.pcolormesh(
        scenario.true_model.x,
        scenario.true_model.z,
        gradient,
        shading="auto",
        cmap="seismic",
    )
    ax.set_title("Градиент функции невязки")
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax, label="∂Φ/∂v")
    fig.tight_layout()
    return fig


def _plot_residual_hist(residuals: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals.flatten(), bins=21, edgecolor="black")
    ax.set_title("Распределение невязки (пред - наблюд.)")
    ax.set_xlabel("время, с")
    ax.set_ylabel("число пар")
    fig.tight_layout()
    return fig


def _plot_misfit(history: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history, marker="o")
    ax.set_title("История невязки ½‖r‖²")
    ax.set_xlabel("итерация")
    ax.set_ylabel("значение")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


tab_inversion, tab_hodograph, tab_guide = st.tabs(
    ["Интерактивная инверсия", "Годографы и метод пристрелки", "Гайд"]
)

with tab_inversion:
    st.markdown(
        """
        **Шаг 1.** Настройте стартовую модель в сайдбаре и зафиксируйте её.
        **Шаг 2.** Нажмите «Сделать шаг», чтобы выполнить одно обновление
        прямолучевой инверсии. Предел поправки задаёт ползунок «Максимальная поправка».
        **Шаг 3.** Экспериментируйте с λ, чтобы увидеть эффект регуляризации.
        """
    )
    col_left, col_right = st.columns(2)
    with col_left:
        st.pyplot(_plot_velocity(scenario.true_model, "Истинная модель", overlay_interfaces=True))
    with col_right:
        st.pyplot(_plot_velocity(current_model, "Текущая оценка"))

    col_grad, col_hist = st.columns(2)
    with col_grad:
        st.pyplot(_plot_gradient(st.session_state.gradient))
    with col_hist:
        st.pyplot(_plot_residual_hist(residual))

    if st.session_state.misfit_history:
        st.pyplot(_plot_misfit(st.session_state.misfit_history))

    with st.expander("Алгоритм одной итерации", expanded=False):
        st.markdown(
            """
            1. Для каждой пары источник‑приёмник интегрируем время вдоль прямого луча.
            2. Сравниваем с наблюдаемыми временами и получаем невязку `r`.
            3. Строим градиент: длина участка луча в ячейке ∝ вкладу в ошибку.
            4. Добавляем регуляризационный член λ‖v − v₀‖², который тянет модель к стартовой.
            5. Масштабируем градиент так, чтобы максимальная поправка не превышала заданную,
               и делаем шаг `v ← v - Δv`, после чего ограничиваем скорости.
            """
        )

with tab_hodograph:
    st.markdown(
        """
        Годограф показывает зависимость времени прихода волны от смещения.
        В двухслойной модели есть прямая и головная волна. Смещение, при котором
        они пересекаются, называют *точкой схождения* (crossover distance).
        """
    )
    hodograph_col, ray_col = st.columns(2)
    with hodograph_col:
        fig_hodo, ax = plt.subplots(figsize=(5, 4))
        ax.plot(hodograph["offsets"], hodograph["direct"], label="Прямая волна v₁")
        if np.any(~np.isnan(hodograph["head"])):
            ax.plot(hodograph["offsets"], hodograph["head"], label="Головная волна v₂")
            ax.axvline(
                hodograph["crossover"], color="gray", linestyle="--", label="Точка схождения"
            )
            ax.axhline(
                hodograph["intercept"], color="gray", linestyle=":", label="Время подхода"
            )
        ax.set_xlabel("Смещение, м")
        ax.set_ylabel("Время прихода, с")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_hodo.tight_layout()
        st.pyplot(fig_hodo)

    with ray_col:
        st.markdown(
            """
            Метод пристрелки: меняем угол выхода луча, пока он не попадёт в приёмник.
            В случае головной волны нужный угол совпадает с критическим — его задаёт
            закон Снеллиуса на границе слоёв.
            """
        )
        try:
            path = shoot_head_wave(layered_model, single_offset)
            target_offset = float(single_offset)
            guess_key = "shoot_guess_along"
            if st.session_state.get("shoot_target_offset") != target_offset:
                st.session_state["shoot_target_offset"] = target_offset
                st.session_state[guess_key] = float(max(target_offset - 2 * horizontal_leg, 0.0))
            guess_along = st.slider(
                "Пробная длина вдоль границы (м)",
                0.0,
                max_offset,
                value=float(st.session_state.get(guess_key, 0.0)),
                step=50.0,
                key=guess_key,
            )
            guess_offset = 2 * horizontal_leg + guess_along
            path_guess = np.array(
                [
                    (0.0, 0.0),
                    (horizontal_leg, depth),
                    (horizontal_leg + guess_along, depth),
                    (guess_offset, 0.0),
                ]
            )

            fig_ray, ax_ray = plt.subplots(figsize=(5, 4))
            ax_ray.plot(path[:, 0], path[:, 1], "-o", label="Критический луч")
            ax_ray.plot(path_guess[:, 0], path_guess[:, 1], "--o", label="Пробный луч")
            ax_ray.axvline(target_offset, color="gray", linestyle=":", label="Цель")
            ax_ray.set_title("Траектория головной волны")
            ax_ray.set_xlabel("x, м")
            ax_ray.set_ylabel("z, м")
            ax_ray.invert_yaxis()
            ax_ray.grid(True, alpha=0.3)
            ax_ray.legend()
            fig_ray.tight_layout()
            st.pyplot(fig_ray)
            delta_offset = guess_offset - target_offset
            st.info(
                f"Пробный луч выходит на поверхность в {guess_offset:.0f} м. "
                f"Отклонение от цели: {delta_offset:+.0f} м."
            )
        except ValueError as exc:
            st.warning(f"Для смещения {single_offset:.0f} м головная волна не возникает: {exc}")

with tab_guide:
    st.markdown(
        """
        ## План объяснения для друга

        **1. Что мы измеряем?**  
        Источники генерируют упругие волны, приёмники фиксируют времена прихода.
        На поверхности измеряем годограф — зависимость времени от расстояния.

        **2. Как представляем среду?**  
        - Для инверсии — сетка скоростей в вертикальном разрезе.  
        - Для аналитики — упрощённая двухслойная модель.  
        - Стартовую модель можно собрать вручную, регулируя скорости слоёв.

        **3. Как моделируем прямую задачу?**  
        В этом демо лучи считаем прямыми. Интегрируем время `T = ∫ ds / v(x, z)`
        вдоль отрезка между источником и приёмником.

        **4. Как ищем модель (обратная задача)?**  
        Используем градиентный спуск. Невязка — половина квадрата разности между
        синтетическими и наблюденными временами, усреднённая по всем лучам.

        **5. Зачем регуляризация?**  
        Данных мало, поэтому штраф λ‖v − v₀‖² предотвращает «рваную» модель и
        удерживает решение рядом со стартовой оценкой.

        **6. Где использовать годограф?**  
        - Быстро оценить глубину и скорость нижнего слоя по точке схождения.  
        - Пояснить, почему дальние приёмники видят головную волну раньше прямой.

        **Практический сценарий занятия**

        1. Настройте параметры годографа, покажите прямую и головную волны.  
        2. Переключитесь на вкладку инверсии: запустите несколько шагов, обсудите,
           как меняется карта скоростей и невязка.  
        3. Повышайте λ, чтобы увидеть, как регуляризация сглаживает модель.  
        4. Вернитесь к графику градиента и объясните, какие зоны влияют чаще всего —
           там проходят большинство лучей.  
        5. Завершите выводами: где модель уверенная, а где нужна дополнительная
           информация (например, наклонные профили, отражённые волны, 3D-съёмка).
        """
    )

st.caption(
    "Код проекта: моделирование в `tomography/`, визуализация в этом файле. "
    "Проверьте README для инструкций по запуску."
)
