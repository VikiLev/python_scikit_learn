{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 186,
   "id": "cbc33734-dd8b-4e6c-9a1d-6eb87e00da0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn import tree\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "0a42d5db-3d41-4bd2-a40a-97a577932566",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Loading data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "2a69c80a-b650-4c97-9e4c-b0539b4eeb04",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Income</th>\n",
       "      <th>Favorite Transport</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>bicycle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>scooter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>bicycle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>16</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>18</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>20</td>\n",
       "      <td>male</td>\n",
       "      <td>200.0</td>\n",
       "      <td>scooter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>22</td>\n",
       "      <td>female</td>\n",
       "      <td>500.0</td>\n",
       "      <td>scooter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>23</td>\n",
       "      <td>male</td>\n",
       "      <td>300.0</td>\n",
       "      <td>scooter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>25</td>\n",
       "      <td>female</td>\n",
       "      <td>800.0</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>27</td>\n",
       "      <td>female</td>\n",
       "      <td>3000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>30</td>\n",
       "      <td>female</td>\n",
       "      <td>5000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>7000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>40</td>\n",
       "      <td>male</td>\n",
       "      <td>9000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>45</td>\n",
       "      <td>male</td>\n",
       "      <td>3000.0</td>\n",
       "      <td>taxi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>47</td>\n",
       "      <td>female</td>\n",
       "      <td>6000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>49</td>\n",
       "      <td>male</td>\n",
       "      <td>8000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>53</td>\n",
       "      <td>female</td>\n",
       "      <td>1300.0</td>\n",
       "      <td>taxi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>56</td>\n",
       "      <td>female</td>\n",
       "      <td>1400.0</td>\n",
       "      <td>taxi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>57</td>\n",
       "      <td>male</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>59</td>\n",
       "      <td>female</td>\n",
       "      <td>5500.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>64</td>\n",
       "      <td>male</td>\n",
       "      <td>6500.0</td>\n",
       "      <td>car</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>68</td>\n",
       "      <td>male</td>\n",
       "      <td>20000.0</td>\n",
       "      <td>helicopter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>70</td>\n",
       "      <td>female</td>\n",
       "      <td>2500.0</td>\n",
       "      <td>taxi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>71</td>\n",
       "      <td>male</td>\n",
       "      <td>25000.0</td>\n",
       "      <td>helicopter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>75</td>\n",
       "      <td>male</td>\n",
       "      <td>30000.0</td>\n",
       "      <td>helicopter</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Age  Gender   Income Favorite Transport\n",
       "0     5  female      NaN            bicycle\n",
       "1     8    male      NaN            scooter\n",
       "2    10  female      NaN            bicycle\n",
       "3    14    male      NaN              metro\n",
       "4    16    male      NaN              metro\n",
       "5    18  female      NaN              metro\n",
       "6    20    male    200.0            scooter\n",
       "7    22  female    500.0            scooter\n",
       "8    23    male    300.0            scooter\n",
       "9    25  female    800.0              metro\n",
       "10   27  female   3000.0                car\n",
       "11   30  female   5000.0                car\n",
       "12   33    male   7000.0                car\n",
       "13   40    male   9000.0                car\n",
       "14   45    male   3000.0               taxi\n",
       "15   47  female   6000.0                car\n",
       "16   49    male   8000.0                car\n",
       "17   53  female   1300.0               taxi\n",
       "18   56  female   1400.0               taxi\n",
       "19   57    male   4000.0                car\n",
       "20   59  female   5500.0                car\n",
       "21   64    male   6500.0                car\n",
       "22   68    male  20000.0         helicopter\n",
       "23   70  female   2500.0               taxi\n",
       "24   71    male  25000.0         helicopter\n",
       "25   75    male  30000.0         helicopter"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('vehicles.csv')\n",
    "df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "b2aa738f-177f-43f9-9b52-d1cddaa3ef85",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Cleaning\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "2f39e922-26aa-46bd-bc5b-34e2035a714b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Age                   0\n",
       "Gender                0\n",
       "Income                6\n",
       "Favorite Transport    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "ce681af5-0e51-4c3d-9ce0-67167c2323b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Income</th>\n",
       "      <th>Favorite Transport</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>female</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bicycle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>scooter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>female</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bicycle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14</td>\n",
       "      <td>male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>16</td>\n",
       "      <td>male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Gender  Income Favorite Transport\n",
       "0    5  female     0.0            bicycle\n",
       "1    8    male     0.0            scooter\n",
       "2   10  female     0.0            bicycle\n",
       "3   14    male     0.0              metro\n",
       "4   16    male     0.0              metro"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Income'].fillna(0.0, inplace=True)\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "add71146-d1bb-4c3b-b3c7-be7aebf2d6ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "7fdbae00-2c8e-4915-a152-e2ed4e491de4",
   "metadata": {},
   "outputs": [],
   "source": [
    "## option 1\n",
    "# df.replace({\n",
    "#     'Gender': {\n",
    "#         'male': 0,\n",
    "#         'female': 1\n",
    "        \n",
    "#     }\n",
    "# }, inplace=True)\n",
    "# df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "249bdb91-7dae-4a5a-a7ec-dfb5f84ea57f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Income</th>\n",
       "      <th>Favorite Transport</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bicycle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>scooter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bicycle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>metro</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Gender  Income Favorite Transport\n",
       "0    5       0     0.0            bicycle\n",
       "1    8       1     0.0            scooter\n",
       "2   10       0     0.0            bicycle\n",
       "3   14       1     0.0              metro\n",
       "4   16       1     0.0              metro"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## option 2\n",
    "label_encoder = LabelEncoder()\n",
    "df['Gender'] = label_encoder.fit_transform(df['Gender'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "30f61b3d-78ec-4045-a5d1-8ce4c4111d4e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Age                     int64\n",
       "Gender                  int64\n",
       "Income                float64\n",
       "Favorite Transport     object\n",
       "dtype: object"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "5be27dc1-5aa4-4c9f-aebe-778f3d3f874a",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "2b35b27f-f189-46b1-99dd-2ee51ab8e54e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Gender  Income\n",
       "0    5       0     0.0\n",
       "1    8       1     0.0\n",
       "2   10       0     0.0\n",
       "3   14       1     0.0\n",
       "4   16       1     0.0"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df.drop(columns='Favorite Transport')\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "bff690c3-5c4d-4b5f-b61c-0f51442f0670",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    bicycle\n",
       "1    scooter\n",
       "2    bicycle\n",
       "3      metro\n",
       "4      metro\n",
       "Name: Favorite Transport, dtype: object"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df['Favorite Transport']\n",
    "y.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "6756e1cf-03db-4409-9ae6-aef3fb736cc0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-13 {color: black;}#sk-container-id-13 pre{padding: 0;}#sk-container-id-13 div.sk-toggleable {background-color: white;}#sk-container-id-13 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-13 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-13 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-13 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-13 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-13 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-13 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-13 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-13 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-13 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-13 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-13 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-13 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-13 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-13 div.sk-item {position: relative;z-index: 1;}#sk-container-id-13 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-13 div.sk-item::before, #sk-container-id-13 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-13 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-13 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-13 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-13 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-13 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-13 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-13 div.sk-label-container {text-align: center;}#sk-container-id-13 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-13 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-13\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" checked><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = DecisionTreeClassifier()\n",
    "model.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "36cb5a40-dac8-439a-92cc-7aa125f3d00e",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "81ecabca-f9b7-4e09-8853-8d321eda4ad4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Gender  Income\n",
       "0   12       0     0.0"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df = pd.DataFrame({\n",
    "    'Age': [12],\n",
    "    'Gender': [0],\n",
    "    'Income': [0.0]\n",
    "})\n",
    "test_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "0aea1f9a-a6e8-42cb-84c0-7b9954d20a66",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['bicycle'], dtype=object)"
      ]
     },
     "execution_count": 174,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(test_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "f3d1e202-cdd2-46bf-bad2-b1135b1bbea0",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "529f3851-4c03-4180-a75f-3a1b9f9dc17c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20, 3)"
      ]
     },
     "execution_count": 176,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
    "X_train.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "ea07a6ad-0416-4701-9862-b3e52bf86dbd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-14 {color: black;}#sk-container-id-14 pre{padding: 0;}#sk-container-id-14 div.sk-toggleable {background-color: white;}#sk-container-id-14 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-14 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-14 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-14 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-14 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-14 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-14 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-14 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-14 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-14 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-14 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-14 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-14 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-14 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-14 div.sk-item {position: relative;z-index: 1;}#sk-container-id-14 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-14 div.sk-item::before, #sk-container-id-14 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-14 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-14 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-14 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-14 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-14 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-14 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-14 div.sk-label-container {text-align: center;}#sk-container-id-14 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-14 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-14\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" checked><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 177,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = DecisionTreeClassifier()\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "51d57cbb-8b39-4e85-9740-d5e27cd3fb13",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>49</td>\n",
       "      <td>1</td>\n",
       "      <td>8000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>71</td>\n",
       "      <td>1</td>\n",
       "      <td>25000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>5000.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Age  Gender   Income\n",
       "16   49       1   8000.0\n",
       "0     5       0      0.0\n",
       "24   71       1  25000.0\n",
       "1     8       1      0.0\n",
       "3    14       1      0.0\n",
       "11   30       0   5000.0"
      ]
     },
     "execution_count": 178,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "49f88c33-fc94-451c-869e-6455a83112e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['car', 'bicycle', 'helicopter', 'bicycle', 'metro', 'car'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictions = model.predict(X_test)\n",
    "predictions\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "eee197a0-94a7-4a2c-ae74-aa3656cb8dec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8333333333333334"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_accuracy_score = accuracy_score(y_test, predictions)\n",
    "model_accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "id": "a9e08f26-2268-4b3b-a704-bd8609025634",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Exporting to the DOT file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "b5f4d818-b621-43b6-b92f-773006bdfda2",
   "metadata": {},
   "outputs": [],
   "source": [
    "tree.export_graphviz(model, out_file='decision_tree_model.dot', feature_names=['Age', 'Gender', 'Income'], filled=True, class_names=sorted(y.unique()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "e2321aca-1a4b-41c6-9d98-667d9b678e45",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Charts\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "d93dffdb-5d50-4fdf-bd63-6368244cda25",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Gender', ylabel='count'>"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAioAAAGwCAYAAACHJU4LAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuNSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/xnp5ZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA5M0lEQVR4nO3deVxVdeL/8fcFZd8UF1ARUNwDFbdhaJRRzGxyciptfJC5O+WSDm7xc1wwS61xDbXGUrJytDK1NLcYxQY31CRMUyNMG1FMEwQFFe7vDx/e79zQSRE4R3k9H4/7eHCWe8773seF++aczz3XYrVarQIAADAhB6MDAAAA3A5FBQAAmBZFBQAAmBZFBQAAmBZFBQAAmBZFBQAAmBZFBQAAmFYVowPci+LiYp0+fVqenp6yWCxGxwEAAHfAarXq0qVLqlOnjhwc/vcxk/u6qJw+fVoBAQFGxwAAAKVw6tQp1atX73+uc18XFU9PT0k3HqiXl5fBaQAAwJ3Izc1VQECA7X38f7mvi8rN0z1eXl4UFQAA7jN3MmyDwbQAAMC0KCoAAMC0KCoAAMC07usxKsCDqqioSNeuXTM6BkzEycnpVz/GCTyIKCqAiVitVp05c0YXL140OgpMxsHBQcHBwXJycjI6ClChKCqAidwsKbVq1ZKbmxsXMoSk/7u4ZVZWlurXr8/rApUKRQUwiaKiIltJ8fX1NToOTKZmzZo6ffq0rl+/rqpVqxodB6gwnPAETOLmmBQ3NzeDk8CMbp7yKSoqMjgJULEoKoDJcFgft8LrApUVRQUAAJgWRQUAAJiWoUVl6tSpslgsdremTZsaGQnAXQgKCtK8efOMjgHgAWb4EZUWLVooKyvLdvv3v/9tdCTAdPr371+i1FssFn333XeG5kpNTdXQoUNt0xaLRWvXrr2nbQYFBd3ysd689e/f/95Cm1BUVJRGjx5tdAzAlAz/eHKVKlXk5+dndAzA9B599FEtW7bMbl7NmjUNyXL16lU5OTmVy/5TU1Ntn2zZuXOnnnrqKR09etT2Demurq5261+7du2+/bjuzecRwO0ZfkTl+PHjqlOnjho0aKCYmBidPHnytusWFhYqNzfX7gZUFs7OzvLz87O7OTo6as6cOQoNDZW7u7sCAgI0bNgw5eXlSZJyc3Pl6uqqjRs32m1rzZo18vT01OXLlyVJ6enp6ty5s1xdXeXr66uhQ4fatiHdOKLTs2dPvfLKK6pTp46aNGkiyf7UT1BQkCTpT3/6kywWi21aktatW6fw8HC5uLioQYMGio+P1/Xr12/5OGvWrGl7fNWrV5ck1apVS35+fiooKJCPj49WrVqlTp06ycXFRR988IHOnz+vPn36qG7dunJzc1NoaKj++c9/2m03KipKL774osaPH6/q1avLz89PU6dOtS23Wq2aOnWq6tevL2dnZ9WpU0cvvviibXlQUJBefvll9enTR+7u7qpbt64WLlxot4+TJ0/qiSeekIeHh7y8vNS7d2+dPXvWtnzq1Klq1aqV3n77bQUHB8vFxUX9+/dXcnKy5s+fbztqdOLEiVs+N0BlZOgRlQ4dOigxMVFNmjRRVlaW4uPj9bvf/U6HDh2Sp6dnifVnzJih+Ph4A5IC5uXg4KAFCxYoODhY33//vYYNG6bx48dr0aJF8vLy0uOPP64VK1aoe/futvt88MEH6tmzp9zc3JSfn69u3bopIiJCqampys7O1uDBgzVixAglJiba7pOUlCQvLy9t3br1ljlSU1NVq1YtLVu2TI8++qgcHR0lSV9++aWee+45LViwQL/73e+UkZFhO100ZcqUUj3ml156SbNnz1br1q3l4uKigoICtWnTRhMmTJCXl5c2bNigvn37qmHDhmrfvr3tfu+++65iY2O1Z88e7dq1S/3791dkZKS6du2q1atXa+7cuVq5cqVatGihM2fOKC0tzW6/r7/+uv7f//t/io+P1+bNmzVq1Cg1btxYXbt2VXFxsa2kJCcn6/r16xo+fLieeeYZbd++3baN7777TqtXr9Ynn3wiR0dHBQYG6tixY3rooYc0bdo0ScYdKTNacsdORkdQpx3JRkfALxhaVP77D2dYWJg6dOigwMBAffjhhxo0aFCJ9ePi4hQbG2ubzs3NVUBAQIVkBYy2fv16eXh42Ka7d++ujz76yG5sQ1BQkKZPn67nn39eixYtkiTFxMSob9++unz5stzc3JSbm6sNGzZozZo1kqQVK1aooKBAy5cvl7u7uyQpISFBPXr00KxZs1S7dm1Jkru7u95+++3bnqq4+ebq4+Njdzo3Pj5eL730kvr16ydJatCggV5++WWNHz++1EVl9OjRevLJJ+3mjR071vbzyJEjtXnzZn344Yd2RSUsLMy2z0aNGikhIUFJSUnq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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=df['Gender'], hue=df['Favorite Transport'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "id": "f5e93dd6-de86-4636-abbc-d4949f0b914c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Income', ylabel='Count'>"
      ]
     },
     "execution_count": 190,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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W7cOffr0wblz59CyZcvylEZERETVSLn2qIwfPx53797FxYsXcfv2bdy+fRuxsbHIycnBhx9++Nzr6d27N3r16gUXFxc0a9YMn376KczMzHDixInylEVERETVTLn2qOzZswcHDhyAq6urNM/NzQ0rV64s98m0xcXF2LRpE+7duwcvL69S2xQWFqKwsFCazsnJKVdfREREVDWUa4+KVquFoaFhifmGhobQarVlWldMTAzMzMxgZGSEMWPGYOvWrXBzcyu17bx582BhYSG97OzsylM+ERERVRHlCiqvv/46JkyYgOvXr0vz/vrrL/z3v/9Ft27dyrSu5s2bIzo6GlFRUfjggw/g7++PuLi4UtsGBQUhOztbeqWlpZWnfCIiIqoiynXoZ8WKFfj3v/8NR0dHaa9GWloaWrVqhR9//LFM61Kr1XB2dgYAtGvXDqdOncKyZcvwzTfflGhrZGQEIyOj8pRMREREVVC5goqdnR3Onj2LAwcO4NKlSwAAV1dX+Pj4vHBBWq1W5zwUIiIiqrnKFFQOHjyIwMBAnDhxAubm5njjjTfwxhtvAACys7PRsmVLfP3113j11Vefa31BQUHo2bMn7O3tcffuXWzYsAGHDx/G3r17y74lREREVO2UKagsXboUAQEBMDc3L7HMwsICo0ePxuLFi587qGRkZGDo0KFIT0+HhYUFPDw8sHfvXin8EBERUc1WpqBy/vx5LFiw4InLu3fvji+++OK51/f999+XpXsiIiKqYcp01c/NmzdLvSz5kdq1a+PWrVsvXBQRERERUMag8tJLLyE2NvaJyy9cuABbW9sXLoqIiIgIKGNQ6dWrF2bNmoWCgoISy/Lz8xEcHIy33nqrwoojIiKimq1M56jMnDkTW7ZsQbNmzRAYGIjmzZsDAC5duoSVK1eiuLgYM2bMqJRCiYiIqOYpU1CxtrbGsWPH8MEHHyAoKAhCCACASqWCr68vVq5cCWtr60oplIiIiGqeMt/wzcHBAbt27cKdO3dw5coVCCHg4uKCunXrVkZ9REREVIOV6860AFC3bl106NChImshIiIi0lGuhxISERER6QODChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESkWgwoREREpFoMKERERKRaDChERESmWrEFl3rx56NChA+rUqYOGDRuib9++SEhIkLMkIiIiUhBZg8qRI0cwbtw4nDhxAvv378f9+/fRvXt33Lt3T86yiIiISCFqy9n5nj17dKbDwsLQsGFDnDlzBp07dy7RvrCwEIWFhdJ0Tk5Opdf4V2YBLNOyK70ffbv+d+GzG1UD8fHxcpdQqaysrGBvby93GURElUbWoPJP2dkPA0G9evVKXT5v3jyEhobqpZasrCwAwIrtKQBS9NKnHO4WFMhdQqW4mZMDFYDBgwfLXUqlMtFoEH/pEsMKEVVbigkqWq0WEydOhLe3N1q1alVqm6CgIEyaNEmazsnJgZ2dXaXUk5eXBwBo3707mjo7VEofcrqSmIwzBw4g/8F9uUupFDn5eRAAPuv9b3Rs0ULucipF4o0bGPvDemRmZjKoEFG1pZigMm7cOMTGxuKPP/54YhsjIyMYGRnpsSqgTt16aNjIVq996sPNW9XvcFZpmlhZwaOSwiwREVU+RQSVwMBA7Ny5E0ePHkXjxo3lLoeIiIgUQtagIoTA+PHjsXXrVhw+fBhOTk5ylkNEREQKI2tQGTduHDZs2IDt27ejTp06uHHjBgDAwsICGo1GztKIiIhIAWS9j8qqVauQnZ2Nrl27wtbWVnr98ssvcpZFRERECiH7oR8iIiKiJ+GzfoiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEYVIiIiEixGFSIiIhIsRhUiIiISLEY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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(x=df['Income'], hue=df['Favorite Transport'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d6de723-0595-40b8-9c68-b9bab3c1686f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
