from typing import List
from .models import Product


def recommend_products(predicted_income: float, payload: dict) -> List[Product]:
    recommendations: List[Product] = []
    income = predicted_income
    credit_score = float(payload.get("credit_score", 0))
    savings = float(payload.get("savings_balance", 0))
    dependents = int(payload.get("dependents", 0))
    employment = str(payload.get("employment_type", "")).lower()

    if income >= 200000 and credit_score >= 750:
        recommendations.append(
            Product(
                product_id="premium_cc",
                name="Премиальная кредитная карта",
                description="Повышенный кешбэк, страхование путешествий, доступ в бизнес-залы.",
                reason="Высокий доход и хороший скоринг позволяют комфортно пользоваться премиальными лимитами.",
            )
        )
    if 100000 <= income < 200000:
        recommendations.append(
            Product(
                product_id="cash_loan",
                name="Кредит наличными",
                description="Быстрый кредит на крупные цели без залога.",
                reason="Доход достаточен для обслуживания умеренного ежемесячного платежа.",
            )
        )
    if income < 100000:
        recommendations.append(
            Product(
                product_id="installment_card",
                name="Карта рассрочки",
                description="Покупки без переплаты у партнеров, контроль расходов.",
                reason="Позволяет распределить траты без увеличения долговой нагрузки.",
            )
        )
    if savings >= 50000:
        recommendations.append(
            Product(
                product_id="investment",
                name="Инвестиционный счет",
                description="Подбор портфеля облигаций и фондов с умеренным риском.",
                reason="Есть накопления, которые можно диверсифицировать для роста.",
            )
        )
    if employment in {"salaried", "self_employed"} and dependents <= 1 and credit_score >= 680:
        recommendations.append(
            Product(
                product_id="credit_card",
                name="Кредитная карта",
                description="Грейс-период до 50 дней и кешбэк на повседневные траты.",
                reason="Подходит для ежедневных расходов при контроле платежной дисциплины.",
            )
        )
    if income >= 150000 and savings >= 200000:
        recommendations.append(
            Product(
                product_id="premium_debit",
                name="Премиальный дебетовый пакет",
                description="Персональный менеджер, повышенные ставки на остаток, скидки партнёров.",
                reason="Высокий доход и остатки на счетах подходят под премиальное обслуживание.",
            )
        )
    if not recommendations:
        recommendations.append(
            Product(
                product_id="basic_account",
                name="Базовый расчетный счет",
                description="Дебетовая карта с бесплатным обслуживанием при минимальных оборотах.",
                reason="Стартовый продукт, пока собираем дополнительные данные о клиенте.",
            )
        )
    return recommendations
