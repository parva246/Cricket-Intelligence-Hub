"""
IPL 2026 Squad Data
====================
Update this file each season with latest squad and availability info.
"""

# Full squads for IPL 2026
SQUADS = {
    'Sunrisers Hyderabad': [
        'Pat Cummins', 'Travis Head', 'Abhishek Sharma', 'Aniket Verma',
        'R Smaran', 'Ishan Kishan', 'Heinrich Klaasen', 'Nitish Kumar Reddy',
        'Harsh Dubey', 'Kamindu Mendis', 'Harshal Patel', 'Brydon Carse',
        'Jaydev Unadkat', 'Eshan Malinga', 'Zeeshan Ansari', 'Shivang Kumar',
        'Salil Arora', 'Sakib Hussain', 'Onkar Tarmale', 'Amit Kumar',
        'Praful Hinge', 'Krains Fuletra', 'Liam Livingstone', 'Shivam Mavi',
        'Jack Edwards'
    ],
    'Chennai Super Kings': [
        'Ruturaj Gaikwad', 'MS Dhoni', 'Sanju Samson', 'Ayush Mhatre',
        'Dewald Brevis', 'Urvil Patel', 'Shivam Dube', 'Jamie Overton',
        'Ramakrishna Ghosh', 'Noor Ahmad', 'Khaleel Ahmed', 'Anshul Kamboj',
        'Gurjapneet Singh', 'Shreyas Gopal', 'Mukesh Choudhary', 'Nathan Ellis',
        'Akeal Hosein', 'Prashant Veer', 'Kartik Sharma', 'Matthew Short',
        'Aman Khan', 'Sarfaraz Khan', 'Matt Henry', 'Rahul Chahar', 'Zak Foulkes'
    ],
    'Mumbai Indians': [
        'Hardik Pandya', 'Rohit Sharma', 'Suryakumar Yadav', 'Tilak Varma',
        'Jasprit Bumrah', 'Trent Boult', 'Quinton de Kock', 'Ryan Rickelton',
        'Robin Minz', 'Raj Bawa', 'Raghu Sharma', 'Mitchell Santner',
        'Corbin Bosch', 'Naman Dhir', 'Allah Ghazanfar', 'Ashwani Kumar',
        'Deepak Chahar', 'Will Jacks', 'Sherfane Rutherford', 'Mayank Markande',
        'Shardul Thakur', 'Reece Topley', 'Satyanarayana', 'Krishnan', 'Luke Wood'
    ],
    'Royal Challengers Bengaluru': [
        'Rajat Patidar', 'Virat Kohli', 'Devdutt Padikkal', 'Phil Salt',
        'Jitesh Sharma', 'Krunal Pandya', 'Swapnil Singh', 'Tim David',
        'Romario Shepherd', 'Jacob Bethell', 'Josh Hazlewood', 'Yash Dayal',
        'Bhuvneshwar Kumar', 'Nuwan Thushara', 'Rasikh Salam', 'Abhinandan Singh',
        'Suyash Sharma', 'Venkatesh Iyer', 'Jacob Duffy', 'Satvik Deswal',
        'Mangesh Yadav', 'Vicky Ostwal', 'Kanishk Chouhan', 'Jordan Cox',
        'Vihaan Malhotra'
    ],
    'Kolkata Knight Riders': [
        'Ajinkya Rahane', 'Rinku Singh', 'Sunil Narine', 'Varun Chakaravarthy',
        'Harshit Rana', 'Ramandeep Singh', 'Angkrish Raghuvanshi', 'Vaibhav Arora',
        'Rovman Powell', 'Manish Pandey', 'Umran Malik', 'Anukul Roy',
        'Cameron Green', 'Matheesha Pathirana', 'Finn Allen', 'Tejasvi Singh',
        'Kartik Tyagi', 'Prashant Solanki', 'Rahul Tripathi', 'Tim Seifert',
        'Sarthak Ranjan', 'Daksh Kamra', 'Mitchell Starc', 'Moeen Ali',
        'Rachin Ravindra'
    ],
    'Lucknow Super Giants': [
        'Rishabh Pant', 'Nicholas Pooran', 'Aiden Markram', 'Abdul Samad',
        'Ayush Badoni', 'Matthew Breetzke', 'Himmat Singh', 'Mitchell Marsh',
        'Shahbaz Ahmed', 'Arshin Kulkarni', 'Mohammad Shami', 'Avesh Khan',
        'Mayank Yadav', 'Mohsin Khan', 'Wanindu Hasaranga', 'Anrich Nortje',
        'Manimaran Siddharth', 'Digvesh Rathi', 'Prince Yadav', 'Akash Singh',
        'Arjun Tendulkar', 'Josh Inglis', 'Akshat Raghuwanshi', 'Mukul Choudhary',
        'Naman Tiwari'
    ],
    'Gujarat Titans': [
        'Shubman Gill', 'Sai Sudharsan', 'Jos Buttler', 'Kumar Kushagra',
        'Anuj Rawat', 'Karim Janat', 'Nishant Sindhu', 'Shahrukh Khan',
        'Rahul Tewatia', 'Washington Sundar', 'Glenn Phillips', 'Arshad Khan',
        'Mohammed Siraj', 'Kagiso Rabada', 'Prasidh Krishna', 'Ishant Sharma',
        'Gurnoor Singh Brar', 'Rashid Khan', 'Sai Kishore', 'Manav Suthar',
        'Jayant Yadav', 'Gerald Coetzee', 'Ashok Sharma', 'Jason Holder',
        'Tom Banton', 'Prithvi Raj Yarra', 'Luke Wood'
    ],
    'Rajasthan Royals': [
        'Riyan Parag', 'Ravindra Jadeja', 'Sam Curran', 'Yashasvi Jaiswal',
        'Dhruv Jurel', 'Shimron Hetmyer', 'Sandeep Sharma', 'Shubham Dubey',
        'Vaibhav Suryavanshi', 'Lhuan-dre Pretorius', 'Yudhvir Singh Charak',
        'Jofra Archer', 'Tushar Deshpande', 'Kwena Maphaka', 'Nandre Burger',
        'Ravi Bishnoi', 'Sushant Mishra', 'Yash Raj Punja', 'Donovan Ferreira',
        'Aman Rao Perala', 'Vignesh Puthur', 'Ravi Singh', 'Adam Milne',
        'Kuldeep Sen', 'Brijesh Sharma'
    ],
    'Delhi Capitals': [
        'KL Rahul', 'Axar Patel', 'Kuldeep Yadav', 'Tristan Stubbs',
        'Abishek Porel', 'Nitish Rana', 'Karun Nair', 'Mitchell Starc',
        'Mukesh Kumar', 'T Natarajan', 'Dushmantha Chameera', 'Sameer Rizvi',
        'Ashosh Sharma', 'Vipraj Nigam', 'Ajay Mandal', 'Tripurana Vijay',
        'Madhav Tiwari', 'David Miller', 'Ben Duckett', 'Pathum Nissanka',
        'Lungi Ngidi', 'Sahil Parakh', 'Prithvi Shaw', 'Kyle Jamieson',
        'Auqib Dar'
    ],
    'Punjab Kings': [
        'Shreyas Iyer', 'Priyansh Arya', 'Pyla Avinash', 'Harnoor Singh',
        'Prabhsimran Singh', 'Vishnu Vinod', 'Nehal Wadhera', 'Azmatullah Omarzai',
        'Cooper Connolly', 'Marco Jansen', 'Musheer Khan', 'Mitchell Owen',
        'Shashank Singh', 'Marcus Stoinis', 'Suryansh Shedge', 'Arshdeep Singh',
        'Xavier Bartlett', 'Yuzvendra Chahal', 'Praveen Dubey', 'Ben Dwarshuis',
        'Lockie Ferguson', 'Harpreet Brar', 'Vishal Nishad', 'Vijaykumar Vyshak',
        'Yash Thakur'
    ]
}

# Player name variations — maps how names appear in Cricsheet data to squad names
# Cricsheet uses initial-based names (e.g., "PJ Cummins" for "Pat Cummins")
NAME_VARIATIONS = {
    # SRH
    'PJ Cummins': 'Pat Cummins',
    'TM Head': 'Travis Head',
    'H Klaasen': 'Heinrich Klaasen',
    'LS Livingstone': 'Liam Livingstone',
    'HV Patel': 'Harshal Patel',
    'S Mavi': 'Shivam Mavi',
    'JD Unadkat': 'Jaydev Unadkat',
    # CSK
    'RD Gaikwad': 'Ruturaj Gaikwad',
    'MS Dhoni': 'MS Dhoni',
    'SV Samson': 'Sanju Samson',
    'S Dube': 'Shivam Dube',
    'D Brevis': 'Dewald Brevis',
    'N Ellis': 'Nathan Ellis',
    'M Henry': 'Matt Henry',
    'RD Chahar': 'Rahul Chahar',
    'S Gopal': 'Shreyas Gopal',
    'KK Ahmed': 'Khaleel Ahmed',
    'A Kamboj': 'Anshul Kamboj',
    'DJM Short': 'Matthew Short',
    'AJ Hosein': 'Akeal Hosein',
    'Mukesh Choudhary': 'Mukesh Choudhary',
    # MI
    'HH Pandya': 'Hardik Pandya',
    'RG Sharma': 'Rohit Sharma',
    'SA Yadav': 'Suryakumar Yadav',
    'SK Yadav': 'Suryakumar Yadav',
    'JJ Bumrah': 'Jasprit Bumrah',
    'TA Boult': 'Trent Boult',
    'T Boult': 'Trent Boult',
    'Q de Kock': 'Quinton de Kock',
    'DL Chahar': 'Deepak Chahar',
    'SN Thakur': 'Shardul Thakur',
    'MJ Santner': 'Mitchell Santner',
    'RA Bawa': 'Raj Bawa',
    'WG Jacks': 'Will Jacks',
    'SE Rutherford': 'Sherfane Rutherford',
    'M Markande': 'Mayank Markande',
    'RJW Topley': 'Reece Topley',
    'L Wood': 'Luke Wood',
    # RCB
    'V Kohli': 'Virat Kohli',
    'RM Patidar': 'Rajat Patidar',
    'D Padikkal': 'Devdutt Padikkal',
    'PD Salt': 'Phil Salt',
    'KH Pandya': 'Krunal Pandya',
    'TH David': 'Tim David',
    'VR Iyer': 'Venkatesh Iyer',
    'JR Hazlewood': 'Josh Hazlewood',
    'B Kumar': 'Bhuvneshwar Kumar',
    'Y Dayal': 'Yash Dayal',
    'R Shepherd': 'Romario Shepherd',
    # KKR
    'AM Rahane': 'Ajinkya Rahane',
    'SP Narine': 'Sunil Narine',
    'CV Varun': 'Varun Chakaravarthy',
    'R Powell': 'Rovman Powell',
    'MA Starc': 'Mitchell Starc',
    'MM Ali': 'Moeen Ali',
    'M Pathirana': 'Matheesha Pathirana',
    'FH Allen': 'Finn Allen',
    'Umran Malik': 'Umran Malik',
    'MK Pandey': 'Manish Pandey',
    'RA Tripathi': 'Rahul Tripathi',
    'TL Seifert': 'Tim Seifert',
    'R Ravindra': 'Rachin Ravindra',
    'C Green': 'Cameron Green',
    'Harshit Rana': 'Harshit Rana',
    # LSG
    'RR Pant': 'Rishabh Pant',
    'N Pooran': 'Nicholas Pooran',
    'AK Markram': 'Aiden Markram',
    'A Badoni': 'Ayush Badoni',
    'MR Marsh': 'Mitchell Marsh',
    'Mohammed Shami': 'Mohammad Shami',
    'Avesh Khan': 'Avesh Khan',
    'A Nortje': 'Anrich Nortje',
    'PWH de Silva': 'Wanindu Hasaranga',
    'M Siddharth': 'Manimaran Siddharth',
    # GT
    'B Sai Sudharsan': 'Sai Sudharsan',
    'JC Buttler': 'Jos Buttler',
    'R Tewatia': 'Rahul Tewatia',
    'GL Phillips': 'Glenn Phillips',
    'K Rabada': 'Kagiso Rabada',
    'PJ Krishna': 'Prasidh Krishna',
    'GD Coetzee': 'Gerald Coetzee',
    'JO Holder': 'Jason Holder',
    'I Sharma': 'Ishant Sharma',
    'M Shahrukh Khan': 'Shahrukh Khan',
    'MJ Suthar': 'Manav Suthar',
    'R Sai Kishore': 'Sai Kishore',
    'T Banton': 'Tom Banton',
    'Gurnoor Brar': 'Gurnoor Singh Brar',
    'Mohammed Siraj': 'Mohammed Siraj',
    'Washington Sundar': 'Washington Sundar',
    'Rashid Khan': 'Rashid Khan',
    'S Gill': 'Shubman Gill',
    # RR
    'R Parag': 'Riyan Parag',
    'RA Jadeja': 'Ravindra Jadeja',
    'SM Curran': 'Sam Curran',
    'YBK Jaiswal': 'Yashasvi Jaiswal',
    'SO Hetmyer': 'Shimron Hetmyer',
    'JC Archer': 'Jofra Archer',
    'TD Deshpande': 'Tushar Deshpande',
    'KT Maphaka': 'Kwena Maphaka',
    'N Burger': 'Nandre Burger',
    'D Ferreira': 'Donovan Ferreira',
    'AF Milne': 'Adam Milne',
    'Ravi Bishnoi': 'Ravi Bishnoi',
    # DC
    'KL Rahul': 'KL Rahul',
    'AR Patel': 'Axar Patel',
    'KK Nair': 'Karun Nair',
    'DA Miller': 'David Miller',
    'PP Shaw': 'Prithvi Shaw',
    'L Ngidi': 'Lungi Ngidi',
    'KA Jamieson': 'Kyle Jamieson',
    'T Stubbs': 'Tristan Stubbs',
    'N Rana': 'Nitish Rana',
    'PVD Chameera': 'Dushmantha Chameera',
    'T Natarajan': 'T Natarajan',
    'Kuldeep Yadav': 'Kuldeep Yadav',
    'Mukesh Kumar': 'Mukesh Kumar',
    'BA Duckett': 'Ben Duckett',
    'PHKD Mendis': 'Pathum Nissanka',
    # PBKS
    'SS Iyer': 'Shreyas Iyer',
    'MP Stoinis': 'Marcus Stoinis',
    'YS Chahal': 'Yuzvendra Chahal',
    'M Jansen': 'Marco Jansen',
    'LH Ferguson': 'Lockie Ferguson',
    'X Bartlett': 'Xavier Bartlett',
    'Harpreet Brar': 'Harpreet Brar',
    'P Simran Singh': 'Prabhsimran Singh',
    'N Wadhera': 'Nehal Wadhera',
    'P Dubey': 'Praveen Dubey',
    'Arshdeep Singh': 'Arshdeep Singh',
    'Musheer Khan': 'Musheer Khan',
}

# Mark players as unavailable (injured/dropped) — update before each match
UNAVAILABLE = {
    # 'Sunrisers Hyderabad': ['Player Name'],
}
