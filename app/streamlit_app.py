"""Streamlit frontend for the credit scorecard API (Russian UI)."""

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000/credit-scorecard/score"

CHECKING_STATUS = {
    "A11": "Овердрафт (< 0 марок)",
    "A12": "0 – 200 марок",
    "A13": "Более 200 марок",
    "A14": "Нет расчётного счёта",
}
CREDIT_HISTORY = {
    "A30": "Кредитов не было / всё погашено",
    "A31": "Все кредиты в этом банке погашены вовремя",
    "A32": "Текущие кредиты погашаются вовремя",
    "A33": "Были просрочки в прошлом",
    "A34": "Проблемный счёт / кредиты в других банках",
}
PURPOSE = {
    "A40": "Автомобиль (новый)",
    "A41": "Автомобиль (б/у)",
    "A42": "Мебель/оборудование",
    "A43": "Радио/телевизор",
    "A44": "Бытовая техника",
    "A45": "Ремонт",
    "A46": "Образование",
    "A48": "Переквалификация",
    "A49": "Бизнес",
    "A410": "Другое",
}
SAVINGS_STATUS = {
    "A61": "Менее 100 марок",
    "A62": "100 - 500 марок",
    "A63": "500 - 1000 марок",
    "A64": "Более 1000 марок",
    "A65": "Неизвестно / нет сберсчёта",
}
EMPLOYMENT = {
    "A71": "Безработный",
    "A72": "Менее 1 года",
    "A73": "1 – 4 года",
    "A74": "4 – 7 лет",
    "A75": "Более 7 лет",
}
PERSONAL_STATUS_SEX = {
    "A91": "Мужчина, в разводе",
    "A92": "Женщина, в разводе или замужем",
    "A93": "Мужчина, холост",
    "A94": "Мужчина, женат или вдовец",
}
OTHER_PARTIES = {
    "A101": "Нет",
    "A102": "Созаёмщик",
    "A103": "Поручитель",
}
PROPERTY_MAGNITUDE = {
    "A121": "Недвижимость",
    "A122": "Накопительный счёт / страхование жизни",
    "A123": "Автомобиль или другое",
    "A124": "Неизвестно / нет",
}
OTHER_PAYMENT_PLANS = {
    "A141": "Банк",
    "A142": "Магазины",
    "A143": "Нет",
}
HOUSING = {
    "A151": "Аренда",
    "A152": "Собственное",
    "A153": "Предоставлено бесплатно",
}
FOREIGN_WORKER = {
    "A201": "Да",
    "A202": "Нет",
}


def _select(label: str, options: dict[str, str]) -> str:
    # Selectbox shows human-readable labels, returns the underlying code
    choice = st.selectbox(label, list(options.values()))
    return next(code for code, text in options.items() if text == choice)


def render_waterfall(breakdown: list[dict]) -> go.Figure:
    """Horizontal waterfall of each feature's scorecard points contribution."""
    df = pd.DataFrame(breakdown).sort_values("points", ascending=True)
    labels = [f"{row['feature']} ({row['bin']})" for _, row in df.iterrows()]
    values = df["points"].tolist()

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(values),
            y=labels,
            x=values,
            connector={"line": {"color": "rgba(150,150,150,0.4)", "width": 1}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Вклад признаков в итоговый балл",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"size": 13},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=500,
    )
    return fig


st.set_page_config(page_title="Кредитный скоринг", page_icon="💳")
st.title("Скоринговая карта для розничного кредитования")
st.caption("Датасет German Credit — модель PD на основе WOE и логистической регрессии")

with st.form("applicant_form"):
    st.subheader("Данные заявителя")

    col1, col2 = st.columns(2)
    with col1:
        checking_status = _select("Статус расчётного счёта", CHECKING_STATUS)
        credit_history = _select("Кредитная история", CREDIT_HISTORY)
        purpose = _select("Цель кредита", PURPOSE)
        savings_status = _select("Статус сберегательного счёта", SAVINGS_STATUS)
        employment = _select("Стаж работы", EMPLOYMENT)
        personal_status_sex = _select("Семейное положение / пол", PERSONAL_STATUS_SEX)
        other_parties = _select("Созаёмщики / поручители", OTHER_PARTIES)
        housing = _select("Жильё", HOUSING)

    with col2:
        property_magnitude = _select("Имущество", PROPERTY_MAGNITUDE)
        other_payment_plans = _select("Другие рассрочки/кредиты", OTHER_PAYMENT_PLANS)
        foreign_worker = _select("Иностранный работник", FOREIGN_WORKER)
        duration = st.number_input(
            "Срок кредита (мес.)", min_value=1, max_value=72, value=12
        )
        credit_amount = st.number_input(
            "Сумма кредита (марок)", min_value=100, max_value=20000, value=2000
        )
        installment_rate = st.slider(
            "Доля платежа от располагаемого дохода (%)", 1, 4, 2
        )
        age = st.number_input("Возраст", min_value=18, max_value=100, value=35)

    submitted = st.form_submit_button("Рассчитать скоринг")

if submitted:
    payload = {
        "checking_status": checking_status,
        "credit_history": credit_history,
        "purpose": purpose,
        "savings_status": savings_status,
        "employment": employment,
        "personal_status_sex": personal_status_sex,
        "other_parties": other_parties,
        "property_magnitude": property_magnitude,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "foreign_worker": foreign_worker,
        "duration": duration,
        "credit_amount": credit_amount,
        "installment_rate": installment_rate,
        "age": age,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Вероятность дефолта (PD)", f"{result['pd']:.2%}")
        col2.metric("Баллы скоринга", f"{result['points']:.1f}")

        band = result["risk_band"]
        band_labels = {"low": "🟢 НИЗКИЙ", "medium": "🟡 СРЕДНИЙ", "high": "🔴 ВЫСОКИЙ"}
        col3.metric("Уровень риска", band_labels[band])

        st.plotly_chart(render_waterfall(result["breakdown"]), use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error(
            "Не удалось подключиться к API. Запущен ли сервер? (`uvicorn api.main:app --reload`)"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"Ошибка API: {e}")
