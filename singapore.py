import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import datetime
import joblib


#set up page configuration for streamlit
st.set_page_config(page_title='Singapore Flat Resale Price Predictor',page_icon='flats',initial_sidebar_state='expanded',layout='wide')

st.markdown("<h1 style='text-align: center; color: white;'>SINGAPORE RESALE PREDICTION</h1>", unsafe_allow_html=True)

st.markdown(f""" <style>.stApp {{
                    position: absolute;
                    width: 100%;
                    overflow: hidden;
                    height: 100%;
                    bottom: -1px;
                    background : url("https://w0.peakpx.com/wallpaper/421/437/HD-wallpaper-plain-purple-background-purple.jpg");
                    background-size: cover}}
                    </style>""",unsafe_allow_html=True) 

#set up the sidebar with optionmenu
selected = st.selectbox("Select an option", ["HOME", "GET PREDICTION"], 
                        format_func=lambda x: f'{x}',
                        help="Choose an option",
                        key=None)

#user input values for selectbox and encoded for respective features
class option:

    option_months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

    encoded_month= {"January" : 1,"February" : 2,"March" : 3,"April" : 4,"May" : 5,"June" : 6,"July" : 7,"August" : 8,"September" : 9,
            "October" : 10 ,"November" : 11,"December" : 12}

    option_town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH','BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
        'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG','SERANGOON',
        'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS','PUNGGOL']
    
    encoded_town={'ANG MO KIO' : 0 ,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,'BUKIT MERAH' : 4,'BUKIT PANJANG' : 5,'BUKIT TIMAH' : 6,
        'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
        'KALLANG/WHAMPOA' : 14,'LIM CHU KANG' : 15,'MARINE PARADE' : 16,'PASIR RIS' : 17,'PUNGGOL' : 18,'QUEENSTOWN' : 19,
        'SEMBAWANG' : 20,'SENGKANG' : 21,'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26}
    
    option_flat_type=['1 ROOM', '2 ROOM','3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']

    encoded_flat_type={'1 ROOM': 0,'2 ROOM' : 1,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,'EXECUTIVE' : 5,'MULTI-GENERATION' : 6}

    option_flat_model=['2-ROOM','3GEN','ADJOINED FLAT', 'APARTMENT' ,'DBSS','IMPROVED' ,'IMPROVED-MAISONETTE', 'MAISONETTE',
                    'MODEL A', 'MODEL A-MAISONETTE','MODEL A2' ,'MULTI GENERATION' ,'NEW GENERATION', 'PREMIUM APARTMENT',
                    'PREMIUM APARTMENT LOFT', 'PREMIUM MAISONETTE','SIMPLIFIED', 'STANDARD','TERRACE','TYPE S1','TYPE S2']

    encoded_flat_model={'2-ROOM' : 0,'3GEN' : 1,'ADJOINED FLAT' : 2,'APARTMENT' : 3,'DBSS' : 4,'IMPROVED' : 5,'IMPROVED-MAISONETTE' : 6,
                'MAISONETTE' : 7,'MODEL A' : 8,'MODEL A-MAISONETTE' : 9,'MODEL A2': 10,'MULTI GENERATION' : 11,'NEW GENERATION' : 12,
                'PREMIUM APARTMENT' : 13,'PREMIUM APARTMENT LOFT' : 14,'PREMIUM MAISONETTE' : 15,'SIMPLIFIED' : 16,'STANDARD' : 17,
                'TERRACE' : 18,'TYPE S1' : 19,'TYPE S2' : 20}
    option_block=['999', '998', '997', '996', '995', '992', '991', '990', '989', '988', '987', '986', '985', '984', '981', '980', '979', '978',
                   '977', '976', '975', '974', '973', '972', '971', '970', '969', '968', '967', '966', '965', '964', '963', '962', '961', '960', 
                   '959', '958', '957', '956', '955', '954', '953', '952', '951', '950', '949', '948', '947', '946', '945', '944', '943', '942',
                    '941', '940', '939', '938', '937', '936', '935', '934', '933', '932', '931', '930', '929', '928', '927', '926', '925', '924', 
                    '923', '922', '921', '920', '919', '918', '917', '916', '915', '914', '913', '912', '911', '910', '909', '908', '907', '906',
                    '905', '904', '903', '902', '899', '898', '897', '896', '895', '894', '893', '892', '891', '890', '889', '888', '887', '886',
                    '885', '884', '883', '882', '881', '880', '879', '878', '877', '876', '875', '874', '873', '872', '871', '870', '869', '868',
                    '867', '866', '865', '864', '863', '862', '861', '860', '859', '858', '857', '856', '855', '854', '853', '852', '851', '850',
                    '849', '848', '847', '846', '845', '844', '843', '842', '841', '840', '839', '838', '837', '836', '835', '834', '833', '832', 
                    '831', '830', '829', '828', '827', '826', '825', '824', '823', '822', '821', '820', '819', '818', '817', '816', '815', '814', 
                    '813', '812', '811', '810', '809', '808', '807', '806', '805', '804', '803', '802', '801', '800', '799', '798', '797', '796', 
                    '795', '794', '793', '792', '791', '790', '789', '788', '787', '786', '785', '784', '783', '782', '781', '780', '779', '778', 
                    '777', '776', '775', '774', '773', '772', '771', '770', '769', '768', '767', '766', '765', '764', '763', '762', '761', '760', 
                    '759', '758', '757', '756', '755', '754', '753', '752', '751', '750', '749', '748', '747', '746', '745', '744', '743', '742', 
                    '741', '740', '739', '738', '737', '736', '735', '734', '733', '732', '731', '730', '729', '728', '727', '726', '725', '724', 
                    '723', '722', '721', '720', '719', '718', '717', '716', '715', '714', '713', '712', '711', '710', '709', '708', '707', '706', 
                    '705', '704', '703', '702', '701', '700', '699', '698', '697', '696', '695', '694', '693', '692', '691', '690', '689', '688', 
                    '687', '686', '685', '684', '683', '682', '681', '680', '679', '678', '677', '676', '675', '674', '673', '672', '671', '670', 
                    '669', '668', '667', '666', '665', '664', '663', '662', '661', '660', '659', '658', '657', '656', '655', '654', '653', '652', 
                    '651', '650', '649', '648', '647', '646', '645', '644', '643', '642', '641', '640', '639', '638', '637', '636', '635', '634', 
                    '633', '632', '631', '630', '629', '628', '627', '626', '625', '624', '623', '622', '621', '620', '619', '618', '617', '616', 
                    '615', '614', '613', '612', '611', '610', '609', '608', '607', '606', '605', '604', '603', '602', '601', '596', '593', '592', 
                    '591', '590', '589', '588', '587', '586', '585', '584', '583', '582', '581', '580', '579', '578', '577', '576', '575', '574', 
                    '573', '572', '571', '570', '569', '568', '567', '566', '565', '564', '563', '562', '561', '560', '559', '558', '557', '556', 
                    '555', '554', '553', '552', '551', '550', '549', '548', '547', '546', '545', '544', '543', '542', '541', '540', '539', '538', 
                    '537', '536', '535', '534', '533', '532', '531', '530', '529', '528', '527', '526', '525', '524', '523', '522', '521', '520', 
                    '519', '518', '517', '516', '515', '514', '513', '512', '511', '510', '509', '508', '507', '506', '505', '504', '503', '502', 
                    '501', '500', '499', '498', '497', '496', '495', '494', '493', '492', '491', '490', '489', '488', '487', '486', '485', '484', 
                    '483', '482', '481', '480', '479', '478', '477', '476', '475', '474', '473', '472', '471', '470', '469', '468', '467', '466', 
                    '465', '464', '463', '462', '461', '460', '459', '458', '457', '456', '455', '454', '453', '452', '451', '450', '449', '448', 
                    '447', '446', '445', '444', '443', '442', '441', '440', '439', '438', '437', '436', '435', '434', '433', '432', '431', '430', 
                    '429', '428', '427', '426', '425', '424', '423', '422', '421', '420', '419', '418', '417', '416', '415', '414', '413', '412', 
                    '411', '410', '409', '408', '407', '406', '405', '404', '403', '402', '401', '399', '398', '397', '396', '395', '394', '393', 
                    '392', '391', '390', '389', '388', '387', '386', '385', '384', '383', '382', '381', '380', '379', '378', '377', '376', '375', 
                    '374', '373', '372', '371', '370', '369', '368', '367', '366', '365', '364', '363', '362', '361', '360', '359', '358', '357', 
                    '356', '355', '354', '353', '352', '351', '350', '349', '348', '347', '346', '345', '344', '343', '342', '341', '340', '339', 
                    '338', '337', '336', '335', '334', '333', '332', '331', '330', '329', '328', '327', '326', '325', '324', '323', '322', '321', 
                    '320', '319', '318', '317', '316', '315', '314', '313', '312', '311', '310', '309', '308', '307', '306', '305', '304', '303', 
                    '302', '301', '300', '299', '298', '297', '296', '295', '294', '293', '292', '291', '290', '289', '288', '287', '286', '285', 
                    '284', '283', '282', '281', '280', '279', '278', '277', '276', '275', '274', '273', '272', '271', '270', '269', '268', '267', 
                    '266', '265', '264', '263', '262', '261', '260', '259', '258', '257', '256', '255', '254', '253', '252', '251', '250', '249', 
                    '248', '247', '246', '245', '244', '243', '242', '241', '240', '239', '238', '237', '236', '235', '234', '233', '232', '231', 
                    '230', '229', '228', '227', '226', '225', '224', '223', '222', '221', '220', '219', '218', '217', '216', '215', '214', '213', 
                    '212', '211', '210', '209', '208', '207', '206', '205', '204', '203', '202', '201', '200', '199', '198', '197', '196', '195', 
                    '194', '193', '192', '191', '190', '189', '188', '187', '186', '185', '184', '183', '182', '181', '180', '179', '178', '177', 
                    '176', '175', '174', '173', '172', '171', '170', '169', '168', '167', '166', '165', '164', '163', '162', '161', '160', '159', 
                    '158', '157', '156', '155', '154', '153', '152', '151', '150', '149', '148', '147', '146', '145', '144', '143', '142', '141', 
                    '140', '139', '138', '137', '136', '135', '134', '133', '132', '131', '130', '129', '128', '127', '126', '125', '124', '123', 
                    '122', '121', '120', '119', '118', '117', '116', '115', '114', '113', '112', '111', '110', '109', '108', '107', '106', '105', 
                    '104', '103', '102', '101', '100', '99', '98', '97', '96', '95', '94', '93', '92', '91', '90', '89', '88', '87', '86', '85', 
                    '84', '83', '82', '81', '80', '79', '78', '77', '76', '75', '74', '73', '72', '71', '70', '69', '68', '67', '66', '65', '64', 
                    '63', '62', '61', '60', '59', '58', '57', '56', '55', '54', '53', '52', '51', '50', '49', '48', '47', '46', '45', '44', '43', 
                    '42', '41', '40', '39', '38', '37', '36', '35', '34', '33', '32', '31', '30', '29', '28', '27', '26', '25', '24', '23', '22', 
                    '21', '20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']

#set up information for the 'get prediction' menu
if selected == "GET PREDICTION":
    st.write('')
    st.markdown("<h5 style=color:orange>To Predict the Resale Price of a Flat, Please Provide the Following Information:",unsafe_allow_html=True)
    st.write('')

    # creted form to get the user input 
    with st.form('prediction'):
        col1, col2 = st.columns(2)
    with col1:
        user_month = st.selectbox(label='Month', options=option.option_months, index=None)
        user_town = st.selectbox(label='Town', options=option.option_town, index=None)
        user_flat_type = st.selectbox(label='Flat Type', options=option.option_flat_type, index=None)
        user_flat_model = st.selectbox(label='Flat Model', options=option.option_flat_model, index=None)
        user_floor_area_sqm = st.text_input(label='Floor area sqm')
        user_price_per_sqm = st.text_input(label='Price Per sqm')

    with col2:
        year = st.text_input(label='Year', max_chars=4)
        block = st.selectbox(label='Block', options=option.option_block, index=None)
        lease_commence_date = st.text_input(label='Year of lease commence', max_chars=4)
        user_remaining_lease = st.text_input(label='Remaining lease')
        c1, c2 = st.columns(2)
        with c1:
            storey_lower = st.number_input(label='Storey start', min_value=1, max_value=50)
        with c2:
            storey_upper = st.number_input(label='Storey end', min_value=1, max_value=51)
        st.markdown('<br>', unsafe_allow_html=True)

        button = st.form_submit_button('PREDICT', use_container_width=True)

        if button:
            with st.spinner("Predicting..."):
                # Check whether user filled all required fields
                if not all([user_month, user_town, user_flat_type, user_flat_model, user_floor_area_sqm, user_price_per_sqm, year, block,
                            lease_commence_date, user_remaining_lease, storey_lower, storey_upper]):
                    st.error("Please fill in all required fields.")
                else:
                    try:
                        # Convert inputs to appropriate types
                        current_year = datetime.datetime.now().year
                        lease_commence_year = int(lease_commence_date)
                        age_of_property = current_year - lease_commence_year
                        user_remaining_year = int(user_remaining_lease)
                        year_int = int(year)
                        current_remaining_lease = user_remaining_year - (current_year - year_int)
                        
                        # Handle user_floor_area_sqm and user_price_per_sqm as floats
                        floor_area_sqm = float(user_floor_area_sqm)
                        price_per_sqm = float(user_price_per_sqm)
                        remaining_lease = float(user_remaining_lease) if user_remaining_lease else None

                        # Encode categorical features
                        month = option.encoded_month[user_month]
                        town = option.encoded_town[user_town]
                        flat_type = option.encoded_flat_type[user_flat_type]
                        flat_model = option.encoded_flat_model[user_flat_model]

                        # Load the model
                        model = joblib.load('Regression_Model.joblib')
                       
                        # Prepare user data for prediction
                        user_data = np.array([[month, town, flat_type, block, floor_area_sqm, flat_model, lease_commence_year, remaining_lease, year_int, storey_lower,
                                               storey_upper, price_per_sqm, current_remaining_lease, age_of_property]])

                        # Predict resale price
                        predict = model.predict(user_data)
                        resale_price = predict[0]  # Assuming the model outputs the price directly

                        # Format resale price in SGD
                        formatted_price = f"S${resale_price:,.2f}"

                        # Display the predicted selling price
                        st.subheader(f"Predicted Resale Price is: :green[{formatted_price}]")
                        
                    except ValueError as e:
                        st.error(f"Error in input values: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

#set up information for 'Home' menu 
if selected == "HOME":
    st.subheader(':orange[Domain :]')
    st.markdown('<h5>Real Estate',unsafe_allow_html=True)

    st.subheader(':orange[Skills & Technologies :]')
    st.markdown('<h5> Python scripting, Data Preprocessing,  EDA, Machine learning, Streamlit ',unsafe_allow_html=True)

    st.subheader(':orange[Overview :]')
    st.markdown('''  <h5>Data Collection and Preprocessing  <br>     
                <li> Data Source : Downloaded historical resale flat data from official HDB sources, 
                covering the period from 1990 to the current date. <br>              
                <li> Initial Cleaning: Handled missing values, corrected inconsistencies, and ensured the data's integrity. <br>           
                <li> Feature Engineering: Enhanced the dataset by creating new features and transforming existing ones to
                better capture the underlying patterns.''',unsafe_allow_html=True)
    
    st.markdown('''  <h5>Data Exploration and Handling <br>     
                <li> Outlier Detection: Identified and handled outliers to ensure the model's robustness. <br>                   
                <li>Categorical Encoding: Encoded categorical features using techniques like Label Encoding to convert 
                them into numerical formats suitable for machine learning algorithms.''',unsafe_allow_html=True)
    
    st.markdown('''  <h5>Model Selection and Training <br>     
                <li> Cross-validated different regression models (e.g., Linear Regression, Random Forest Regressor, etc.) <br>              
                <li> Evaluated performance metrics to choose the best model for predicting resale price. <br>           
                <li> Selected the RandomForest Regressor based on its superior performance in terms of R-squared 
                and Mean Squared Error metrics.''',unsafe_allow_html=True)
    
    st.markdown('''  <h5>Model Deployment <br>     
                <li> Model Serialization: Saved the trained RandomForest Regressor model using joblib for later use in the application. <br>              
                <li> Dashboard Development: Built an interactive dashboard using Streamlit to allow users to input 
                relevant features and get predictions on flat resale prices.<br>''',unsafe_allow_html=True)