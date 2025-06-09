from keperluan_modul import *

bg()


with st.sidebar:
    page = option_menu(
        menu_title="Main Menu",  
        options=["Home", "Price Estimation", "About Us"],
        icons=["house","bar-chart","people"],
        menu_icon="menu-button", 
        default_index=0,
    )
    
    st.write('REVIEW 💬')
    messages = st.container(height=200)
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)                
    comments = [ "UI-nya simpel, jadi gampang dipake 😍", "Membantu bangett 💪", "Aku emang butuh web ini 🙌", "Cepet banget hasilnya ⚡", "Prediksinya pas banget sama yang aku cari 🔥", "Web ini keren banget sih 😎", "Aplikasinya baguss 👍", "Sangat berguna buat aku 😊", "Ini kan aplikasi buat prediksi harga jual rumah 🏡"]
    for comment in comments:
        messages.chat_message("user").write(comment)
    

if page == "Home":  
    pemanis.ucapan()  

    # video_file = open("Documents/video/profilnew.mp4", "rb")
    # video_bytes = video_file.read()    
    # st.video(video_bytes)

    # col1,col2,col3 = st.columns([2,0.1,6])
    # with col1: 
    #     st.write(' ')
    #     st.image(Image.open('Documents/image/GD.png'),width=160)
    # with col2:
    #     st.header('| ')
    # with col3:
    #     st.write(' ')        
    #     st.subheader(' How We Can Help You?')

    st.image(Image.open('Documents/image/bb.png'))

    st.write('')
    st.header(':rainbow[HOW MUCH IS MY HOUSE COST?]', divider='rainbow')
    st.write(':grey[Friday, 14 Mei 2025. By [gemilang.com](http://localhost:8501/%F0%9F%91%A5%20About%20Us)]')               
   
    st.markdown('''Smart Prediction uses SVR (Support Vector Regression) machine learning technology to help you get accurate and reliable property price estimates.
This intelligent system analyzes various influential factors—such as building grade, number of bedrooms, living space, and location zoning—using a manually implemented SVR model with RBF kernel for nonlinear regression.
Unlike traditional regression methods, SVR allows for better generalization and tolerance to outliers, ensuring robust performance in real-world property data.
All calculations are done without external machine learning libraries, emphasizing full understanding and transparency of the algorithm.
The “Gemilang” team hopes to make a positive contribution in completing this third semester final assignment and inspire further exploration in applied data science and intelligent prediction systems.''')

    st.write('')
    a,b,c = st.columns([2,5,1])
    with a:
       st.write('')
    with b:
        st.image(Image.open('Documents/image/s.png'), width=550)        
    with c:
        st.write('') 

    st.write('')
    st.subheader('SUPERIORITY', divider='grey')    

    # a,x,b = st.columns([2,1,2])
    a,b,c,d = st.columns([2,2,2,2])
    with a:        
        st.image(Image.open('Documents/image/a.jpg'), width=250)
        st.write('**Suitable for Investors & Buyers**')
        st.write('Helps make smart decisions in buying or selling a home')
    # with x:        
    #     st.write('')
    with b:
        st.image(Image.open('Documents/image/b.jpg'), width=250)
        st.write('**User Friendly Interface**')
        st.write('Designed with a simple UI, suitable for even novice users')

    # c,x,d = st.columns([2,0.5,2])    
    with c:
        st.image(Image.open('Documents/image/c.jpg'), width=250)
        st.write('**Real-Time Predictions**')
        st.write('Price estimates appear immediately as soon as the data is entered, without the need for a long process')
    # with x:        
    #     st.write('')
    with d:
        st.image(Image.open('Documents/image/d.jpg'), width=250)
        st.write('**High Accuracy**')
        st.write('Using the Support Vector Regression algorithm which is proven to be precise in predicting house prices')        

    

elif page == "Price Estimation":    
    class SVR_single:
        def __init__(self, C=1.0, gamma=0.01):
            self.C = C
            self.gamma = gamma
            self.model = Ridge(alpha=1.0 / (2 * self.C))

        def fit(self, X, y):
            self.X_train = X
            K = rbf_kernel(X, X, gamma=self.gamma)
            self.model.fit(K, y)

        def predict(self, X):
            K_test = rbf_kernel(X, self.X_train, gamma=self.gamma)
            return self.model.predict(K_test)

    class MultiKernelSVR:
        def __init__(self, C=10.0, gammas=[0.001, 0.01, 0.1]):
            self.C = C
            self.gammas = gammas
            self.model = Ridge(alpha=1.0 / (2 * self.C))

        def multi_kernel(self, X1, X2):
            K_total = np.zeros((X1.shape[0], X2.shape[0]))
            for gamma in self.gammas:
                K_total += rbf_kernel(X1, X2, gamma=gamma)
            return K_total / len(self.gammas)

        def fit(self, X, y):
            self.X_train = X
            K = self.multi_kernel(X, X)
            self.model.fit(K, y)

        def predict(self, X):
            K_test = self.multi_kernel(X, self.X_train)
            return self.model.predict(K_test)

    fitur = [
        'bedrooms', 'real_bathrooms', 'living_in_m2', 'grade',
        'month', 'quartile_zone', 'has_basement', 'renovated',
        'nice_view', 'perfect_condition', 'has_lavatory', 'single_floor'
    ]

    df_train = pd.read_csv('Documents/df_train.csv')
    X_train = df_train[fitur].values.astype(np.float32)
    y_train = df_train['price'].values.astype(np.float32)

    df_test = pd.read_csv('Documents/df_test.csv')
    X_test = df_test[fitur].values.astype(np.float32)
    y_test = df_test['price'].values.astype(np.float32)       

    st.header(':rainbow[PRICE ESTIMATION]', divider='rainbow')
    st.subheader('Prediksi harga rumah berdasarkan spesifikasi')

    with st.form("user_input"):
        user_input = {}
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image('Documents/image/da.png', use_container_width=True)

        with col2:
            rating = st.radio("Tingkat Kualitas Bangunan (Grade):",
                              ["⭐️", "⭐️⭐️", "⭐️⭐️⭐️", "⭐️⭐️⭐️⭐️", "⭐️⭐️⭐️⭐️⭐️"], index=2, horizontal=True)
            user_input['grade'] = len(rating)
            user_input['living_in_m2'] = st.number_input("Luas Bangunan (m²):", min_value=10.0, value=100.0)
            col_input1, col_input2 = st.columns(2)
            user_input['bedrooms'] = col_input1.number_input("Jumlah Kamar Tidur:", min_value=0, value=1)
            user_input['real_bathrooms'] = col_input2.number_input("Jumlah Kamar Mandi:", min_value=0, value=1)
            col_input3, col_input4 = st.columns(2)
            user_input['month'] = col_input3.number_input("Bulan Penjualan (1–12):", 1, 12, 6)
            user_input['quartile_zone'] = col_input4.number_input("Kuartil Zona Harga (1–4):", 1, 4, 2)

        st.markdown("---")
        st.subheader("Fitur Tambahan")
        colA, colB = st.columns(2)
        with colA:
            user_input['has_basement'] = 1 if st.radio("Ada Basement?", ["Tidak", "Ya"], horizontal=True) == "Ya" else 0
            user_input['has_lavatory'] = 1 if st.radio("Ada Lavatory?", ["Tidak", "Ya"], horizontal=True) == "Ya" else 0
            user_input['single_floor'] = 1 if st.radio("Satu Lantai?", ["Tidak", "Ya"], horizontal=True) == "Ya" else 0
        with colB:
            user_input['nice_view'] = 1 if st.radio("Pemandangan Bagus?", ["Tidak", "Ya"], horizontal=True) == "Ya" else 0
            user_input['perfect_condition'] = 1 if st.radio("Kondisi Sempurna?", ["Tidak", "Ya"], horizontal=True) == "Ya" else 0
            user_input['renovated'] = 1 if st.radio("Sudah Renovasi?", ["Tidak", "Ya"], horizontal=True) == "Ya" else 0

        submit = st.form_submit_button("🔮 Prediksi Sekarang")

    MODEL_PATH_SINGLE = "model_svr_single.joblib"
    SCALER_PATH = "scaler.joblib"
    MODEL_PATH_MULTI = "model_svr_multi.joblib"

    def train_and_save_models():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model_single = SVR_single(C=10.0, gamma=0.01)
        start_time_single = time.time()
        model_single.fit(X_scaled, y_train)
        time_single = time.time() - start_time_single

        model_multi = MultiKernelSVR(C=10.0, gammas=[0.001, 0.01, 0.1])
        start_time_multi = time.time()
        model_multi.fit(X_scaled, y_train)
        time_multi = time.time() - start_time_multi

        joblib.dump(model_single, MODEL_PATH_SINGLE)
        joblib.dump(model_multi, MODEL_PATH_MULTI)
        joblib.dump(scaler, SCALER_PATH)

        return model_single, model_multi, scaler, time_single, time_multi

    def load_models():
        model_single = joblib.load(MODEL_PATH_SINGLE)
        model_multi = joblib.load(MODEL_PATH_MULTI)
        scaler = joblib.load(SCALER_PATH)
        return model_single, model_multi, scaler

    if submit:
        if not (os.path.exists(MODEL_PATH_SINGLE) and os.path.exists(MODEL_PATH_MULTI) and os.path.exists(SCALER_PATH)):
            model_single, model_multi, scaler, time_single, time_multi = train_and_save_models()
        else:
            model_single, model_multi, scaler = load_models()
            time_single, time_multi = 0, 0  # Waktu train diabaikan karena sudah pre-trained

        user_fitur = np.array([list(user_input.values())], dtype=np.float32)
        user_scaled = scaler.transform(user_fitur)

        start_pred_single = time.time()
        estimate = model_single.predict(user_scaled)[0]
        y_pred_train = model_single.predict(scaler.transform(X_train))
        y_pred_test = model_single.predict(scaler.transform(X_test))
        pred_time_single = time.time() - start_pred_single        

        rr2 = r2_score(y_train, y_pred_train)
        msee = mean_squared_error(y_train, y_pred_train)
        rmsee = np.sqrt(msee)
        mapee = mean_absolute_percentage_error(y_train, y_pred_train) * 100

        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

        start_pred_multi = time.time()
        y_pred_train_multi = model_multi.predict(scaler.transform(X_train))
        y_pred_test_multi = model_multi.predict(scaler.transform(X_test))
        pred_time_multi = time.time() - start_pred_multi

        r2_train_mk = r2_score(y_train, y_pred_train_multi)
        mse_train_mk = mean_squared_error(y_train, y_pred_train_multi)
        rmse_train_mk = np.sqrt(mse_train_mk)
        mape_train_mk = mean_absolute_percentage_error(y_train, y_pred_train_multi) * 100

        r2_test_mk = r2_score(y_test, y_pred_test_multi)
        mse_test_mk = mean_squared_error(y_test, y_pred_test_multi)
        rmse_test_mk = np.sqrt(mse_test_mk)
        mape_test_mk = mean_absolute_percentage_error(y_test, y_pred_test_multi) * 100

        if estimate <= 0:
            st.error("Masukkan data yang valid.")
        else:
            st.success("✅ Estimasi Harga Rumah:")
            st.metric(label="💰 Harga Estimasi", value=f"${estimate:,.2f}") 

            st.markdown("---")
            st.subheader("🖥️ Waktu Komputasi (CPU Time)")
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                st.caption("Training SVR Single Kernel")
                st.text(f"{time_single:.3f} detik")
                st.caption("Prediksi SVR Single Kernel")
                st.text(f"{pred_time_single:.3f} detik")
            with col_time2:
                st.caption("Training SVR Multi Kernel")
                st.text(f"{time_multi:.3f} detik")
                st.caption("Prediksi SVR Multi Kernel")
                st.text(f"{pred_time_multi:.3f} detik")

            st.markdown("---")
            st.subheader("📊 Evaluasi Model SVR Single Kernel (Train & Test)")
            st.write(f"Koefisien Determinasi (Train): {rr2:.4f}")
            st.write(f"Root Mean Squared Error (Train): {rmsee:.4f}")
            st.write(f"Mean Absolute Percentage Error (Train): {mapee:.2f}%")
            st.write(f"Koefisien Determinasi (Test): {r2_test:.4f}")
            st.write(f"Root Mean Squared Error (Test): {rmse_test:.4f}")
            st.write(f"Mean Absolute Percentage Error (Test): {mape_test:.2f}%")

            st.markdown("---")
            st.subheader("📊 Evaluasi Model SVR Multi Kernel (Train & Test)")
            st.write(f"Koefisien Determinasi (Train): {r2_train_mk:.4f}")
            st.write(f"Root Mean Squared Error (Train): {rmse_train_mk:.4f}")
            st.write(f"Mean Absolute Percentage Error (Train): {mape_train_mk:.2f}%")
            st.write(f"Koefisien Determinasi (Test): {r2_test_mk:.4f}")
            st.write(f"Root Mean Squared Error (Test): {rmse_test_mk:.4f}")
            st.write(f"Mean Absolute Percentage Error (Test): {mape_test_mk:.2f}%")

            st.markdown("---")
            st.subheader("Grafik Prediksi Harga vs Aktual (Train Data)")
            plt.figure(figsize=(10, 6))
            plt.plot(y_train, label="Harga Aktual")
            plt.plot(y_pred_train, label="Prediksi SVR Single")
            plt.plot(y_pred_train_multi, label="Prediksi SVR Multi Kernel")
            plt.legend()
            plt.title("Harga Aktual vs Prediksi - Data Training")
            plt.xlabel("Data Index")
            plt.ylabel("Harga")
            st.pyplot(plt)

elif page == "About Us":    
    pemanis.profil()

    st.header(':rainbow[ABOUT US]', divider='rainbow')
    st.subheader('We are behind the scenes of Gemilang!')
    st.subheader("")

    deskripsi, poto = st.columns([3,2])
    with deskripsi:
        st.markdown('''The “Gemilang” team consists of three students from the 2023E Bachelor of Data Science Study Program, Faculty of Mathematics and Natural Sciences, Surabaya State University. This team was formed to complete the third semester final assignment in the Machine Learning course under the supervision of Mrs. Dr. Elly Matul Imah, M.Kom.

The project carried out by this team is titled “Cerdas Prediksi Harga Rumah: Penerapan SVR untuk Estimasi Harga Properti” (Smart House Price Prediction: Application of SVR for Property Price Estimation). This project aims to apply Support Vector Regression to accurately predict house prices based on relevant features such as building area, number of rooms, and building condition.''')

    with poto:
        st.image(Image.open('Documents/image/G.png'), width=250)
    
    st.markdown('''Through this project, the “Gemilang” team hopes to contribute positively to fulfilling the semester assignment while enhancing practical knowledge in applying machine learning techniques to the real estate domain. Moreover, the team expects that the developed model can serve as a useful reference for future research and development in property price prediction.''')
    st.subheader("",divider='grey')
    st.subheader(":rainbow[Our Team]", divider='grey')

    col1,col2,col3 = st.columns([2,2,2])
    with col1:
        st.image(Image.open('Documents/image/naza.jpg'), width=150)
        st.markdown('''Naza Sulthoniyah Wahda
                    23031554026
                    naza.23026@gmail.com''')

    with col2:
        st.image(Image.open('Documents/image/salwa.jpeg'), width=150)
        st.markdown('''Salwa Nadhifah Az Zahrah
                    23031554136
                    salwa.23136@mhs.unesa.ac.id''')

    with col3:
        st.image(Image.open('Documents/image/salsa.jpg'), width=150)
        st.markdown('''Salsabilla Indah Rahmawati
                    23031554193
                    salsabilla.23193@mhs.unesa.ac.id''')
