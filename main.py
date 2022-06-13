import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup


st.title('HW')
df = pd.read_csv("all_v2.csv")

page = st.selectbox("Choose your page", ["Визуализация и работа с данными", "Просмотр пробок", "Лучшие жилые комплексы"])

if page == "Просмотр пробок":
    st.header("Просмотр пробок")
    st.markdown("К примеру: 37 и 55.12")
    lan = st.text_input('Введите широту')
    lot = st.text_input('Введите долготу')
    if lan and lot:
        response = requests.get(f"https://static-maps.yandex.ru/1.x/?ll={lan},{lot}&spn=0.1,0.1&l=map,trf")
    else:
        lan = 37.620070
        lot = 55.753630
        response = requests.get(f"https://static-maps.yandex.ru/1.x/?ll={lan},{lot}&spn=0.1,0.1&l=map,trf")
    try:
        img = Image.open(BytesIO(response.content))
        st.image(img)
    except:
        st.error(f"Неверные данные для ввода\n{response.content}")

elif page == "Лучшие жилые комплексы":
    st.title('Лучшие жилые комплексы')
    st.markdown('Получение лучших жилиых комплексов парсером и отобрадение их в датафрейме')
    url = "https://novostroev.ru/novostroyki/rating/moskva/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    names = soup.findAll('a', class_='zhk-rating-item__name link-n-2')
    ratings = soup.findAll('div', class_='rating-badge__value')
    pd_names = []
    pd_user_bests = []
    pd_price_policity = []
    pd_quality = []
    pd_reliability = []
    for i in range(len(names)):
        data = names[i]
        pd_names.append(names[i].text.strip('\n'))
        pd_user_bests.append(float(ratings[i].text.strip('\n')))
        pd_price_policity.append(float(ratings[i+1].text.strip('\n')))
        pd_quality.append(float(ratings[i+2].text.strip('\n')))
        pd_reliability.append(float(ratings[i+3].text.strip('\n')))

    d = {'name': pd_names,
         'best_by_users': pd_user_bests,
         'best_prices': pd_price_policity,
         'best_quality': pd_quality,
         'best_reability': pd_quality}
    df = pd.DataFrame(data=d)
    st.dataframe(df)

elif page == "Визуализация и работа с данными":
    st.title('Визуализация данных')
    st.write(df.head(10))

    # Цена
    st.subheader("Цена")
    fig = plt.figure(figsize=(10, 10))
    sns.distplot(df['price'])
    st.write(fig)

    # Тип домов и их средняя цена
    st.subheader("Тип домов и их средняя цена")
    means = []
    means.append(df[df['building_type'] == 0].price.mean())
    means.append(df[df['building_type'] == 1].price.mean())
    means.append(df[df['building_type'] == 2].price.mean())
    means.append(df[df['building_type'] == 3].price.mean())
    means.append(df[df['building_type'] == 4].price.mean())
    means.append(df[df['building_type'] == 5].price.mean())
    names = ['Другой', 'Панельный', 'Монолитный', 'Кирпичный', 'Блочный', 'Деревянный']
    fig = plt.figure(figsize=(10, 10))
    fig2 = sns.barplot(names, means)
    fig2.set(xlabel='Тип', ylabel='Средняя цена')
    st.write(fig)

    st.title('Обработка датасета')
    """
    Очистка по площади
    """
    st.subheader('Удаление выбросов по площади квартиры')
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(df['area'].plot(kind='box'))
    st.pyplot(fig)
    st.markdown('На графике можно заметить выбросы. От них и нужно избавиться.\nДля этого берем интервал от 0.05 до 0.95 '
                'квантиля, остальные данные убираем')

    code = """
    q1 = df['area'].quantile(.05)
    q3 = df['area'].quantile(.95)
    df = df[df['area'].between(q1, q3, inclusive=True)].copy()
    """
    st.code(code, language='python')

    q1 = df['area'].quantile(.05)
    q3 = df['area'].quantile(.95)
    df = df[df['area'].between(q1, q3, inclusive=True)].copy()

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(df['area'].plot(kind='box'))
    st.pyplot(fig)

    """
    Очистка по цене
    """
    st.subheader('Удаление выбросов по цене квартиры')

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(df['price'].plot(kind='box'))
    st.pyplot(fig)

    st.markdown("Сначала возьмем все цены по модулю")
    code = """
    df['price'] = df.price.abs()
    """
    df['price'] = df.price.abs()

    st.code(code, language='python')
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(df['price'].plot(kind='box'))
    st.pyplot(fig)

    st.markdown("Теперь удалим все выбросы, взяв интервал от 0.1 до 0.95 квантиля")
    code = """
    q1 = df['price'].quantile(.1)
    q3 = df['price'].quantile(.95)
    df = df[df['price'].between(q1, q3, inclusive=True)].copy()
    """
    st.code(code, language='python')
    q1 = df['price'].quantile(.1)
    q3 = df['price'].quantile(.95)
    df = df[df['price'].between(q1, q3, inclusive=True)].copy()
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(df['price'].plot(kind='box'))
    st.pyplot(fig)

    st.title('Регресионные модели')
    st.subheader('Обучение')
    code = """
    tmp = df.copy()
    y = df.price
    tmp.drop('price', axis = 1, inplace = True)
    x = tmp
    polynomial_features= PolynomialFeatures(degree=2)
    xp = polynomial_features.fit_transform(x)
    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)
    """
    st.code(code, language='python')

    df.drop(['date'], axis = 1, inplace=True)
    df.drop(['time'], axis = 1, inplace=True)

    st.markdown("Тут вашему ПК может не хватить мощности, поэтому для дальшей работы программы, можете закоментировать "
                "часть кода")
    # От сюда
    tmp = df.copy()
    y = df.price
    tmp.drop('price', axis = 1, inplace = True)
    x = tmp
    polynomial_features= PolynomialFeatures(degree=2)
    xp = polynomial_features.fit_transform(x)
    # До сюда

    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)
    model.summary()
    st.text(model.summary())
    st.subheader('Вывод:')
    st.markdown("R^2 = 0.546. Из этого можно сделать вывод,\n что наша модель с 55% вероятностью предсказывать реальную "
                "стоимость квартиры.")


