# Прогнозирование оттока клиентов (Customer Churn Prediction)

## Обзор

Проект посвящён прогнозированию оттока клиентов интернет-провайдера с использованием методов машинного обучения.

Цель: выявлять клиентов с высоким риском ухода и анализировать ключевые факторы, влияющие на риск ухода.

В проекте сделан акцент на  анализ данных, feature engineering.

---

## Постановка задачи

Отток клиентов напрямую влияет на выручку компании.

Необходимо построить модель бинарной классификации, которая предсказывает вероятность ухода клиента на основе данных о поведении, контракте и использовании сервисов.

---

## Данные

Датасет содержит обезличенную информацию по клиентам, включая:

* параметры контракта и оставшийся срок действия;
* статистику использования услуг (download / upload);
* подключённые дополнительные сервисы (TV, movie-пакеты);
* биллинг и превышение лимитов.

Целевая переменная:

* `churn` — бинарный признак оттока клиента.

---

## Ключевые сложности

* **Пропуски в контрактных признаках**

  Пропуски в `remaining_contract` были проанализированы и интерпретированы как  **MNAR-пропуски** , отражающие отдельный бизнес-сигнал, а не отсутствующие данные.
* **Корреляция признаков**

  Использование трафика и связанных метрик требовало проверки на мультиколлинеарность.

## Этапы работы

### 1. Разведочный анализ данных (EDA)

* Анализ распределений числовых и категориальных признаков
* Сравнение churn-rate между сегментами клиентов
* Анализ пропусков и проверка гипотезы MNAR

### 2. Feature Engineering

* Добавление индикатора отсутствия контракта
* Масштабирование числовых признаков
* One-Hot Encoding категориальных переменных

### 3. Моделирование

* End-to-end ML pipeline на `scikit-learn`
* Рассмотренные модели:
  * Logistic Regression
  * CatBoost Classifier
* Подбор гиперпараметров и кросс-валидация

### 4. Оценка качества

Использовались метрики:

* ROC AUC
* Precision / Recall
* F1-score
* Confusion Matrix

---

## Результаты

* Достигнуто качество **ROC AUC > 0.98, F1-Score > 0.95 при PR-threshold 0.45** (наилучший результат показал CatBoost)
* Выявлены ключевые факторы оттока:
  * отсутствие долгосрочного контракта;
  * превышение лимитов трафика;
  * низкая вовлечённость в дополнительные сервисы;
* Итоговая модель подходит для  **раннего выявления оттока и retention-кампаний** .

---

## Технологический стек

* Python
* Pandas, NumPy
* Scikit-learn
* CatBoost
* Matplotlib, Seaborn

---

## Структура проекта

<pre class="overflow-visible! px-0!" data-start="2790" data-end="2927"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>.
├── data/
│   └── internet_service_churn.csv
├──Customer_churn.ipynb
├── README.md
└── requirements.txt
</span></span></code></div></div></pre>

---

## Как запустить проект

<pre class="overflow-visible! px-0!" data-start="2958" data-end="3049"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
notebooks/Customer_churn.ipynb
</span></span></code></div></div></pre>

---

## Возможные улучшения

* Добавить более строгую валидацию (Stratified K-Fold) для оценки стабильности качества модели.
* Провести более глубокий анализ важности признаков (SHAP / permutation importance) для усиления бизнес-интерпретации результатов.
* Добавить калибровку вероятностей.
* Расширить feature engineering за счёт относительных и агрегированных признаков потребления услуг.

---

## Автор

**Артур Алеев**

Data Scientist
