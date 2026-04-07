import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Dropout, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── 1. Adatbetöltés és feature engineering ────────────────────────────────────

kivetelek_data = pd.read_csv('data/ATM_OK3.csv', sep=';', usecols=lambda column: not column.startswith('Unnamed'))
kivetelek_data['DATUM_IDO'] = pd.to_datetime(kivetelek_data['DATUM'] + ' ' + kivetelek_data['IDO'])
kivetelek_data = kivetelek_data.drop(columns=['DATUM', 'IDO'])
kivetelek_data['OSSZEG'] = pd.to_numeric(kivetelek_data['OSSZEG'], errors='coerce')
kivetelek_data = kivetelek_data.dropna(subset=['OSSZEG'])

holidays = {"01-01", "03-15", "05-01", "08-20", "10-23", "11-01", "12-25", "12-26", "04-01", "05-20"}
kivetelek_data['IS_HOLIDAY'] = kivetelek_data['DATUM_IDO'].dt.strftime('%m-%d').isin(holidays).astype(int)
kivetelek_data['HOUR'] = kivetelek_data['DATUM_IDO'].dt.hour
kivetelek_data['WEEKDAY'] = kivetelek_data['DATUM_IDO'].dt.weekday
kivetelek_data['IS_WEEKEND'] = (kivetelek_data['WEEKDAY'] >= 5).astype(int)
kivetelek_data['DAY_OF_MONTH'] = kivetelek_data['DATUM_IDO'].dt.day
kivetelek_data['IS_MONTH_END'] = kivetelek_data['DATUM_IDO'].dt.is_month_end.astype(int)

print(f"Nyers adatok: {len(kivetelek_data)} tranzakció")

# ── 2. Aggregálás 4 órás intervallumra ───────────────────────────────────────

agg = '4h'
aggregated_data = (
    kivetelek_data.set_index('DATUM_IDO')
    .groupby('ATM ID')
    .resample(agg)
    .agg({
        'OSSZEG': 'sum',
        'HOUR': 'first',
        'WEEKDAY': 'first',
        'IS_WEEKEND': 'first',
        'DAY_OF_MONTH': 'first',
        'IS_HOLIDAY': 'max',
        'IS_MONTH_END': 'max'
    })
    .reset_index()
)

aggregated_data['ATM_AVG'] = aggregated_data.groupby('DATUM_IDO')['OSSZEG'].transform(
    lambda x: (x.sum() - x) / (len(x) - 1)
).fillna(0)

aggregated_data = aggregated_data.sort_values(['ATM ID', 'DATUM_IDO'])

print(f"Aggregált adatok: {len(aggregated_data)} rekord ({agg} intervallum)")

# ── 3. PACF-alapú lag feature-ök ─────────────────────────────────────────────

look_back = 42
pacf_lags = [126, 43, 47]

for lag in pacf_lags:
    lag_name = f"LAG_{lag}"
    aggregated_data[lag_name] = aggregated_data.groupby('ATM ID')['OSSZEG'].shift(lag)

aggregated_data['DAILY_AVG'] = aggregated_data.groupby('ATM ID')['OSSZEG'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
aggregated_data['DAILY_AVG_1_WEEK_AGO'] = aggregated_data.groupby('ATM ID')['DAILY_AVG'].shift(43)
aggregated_data['WEEKLY_AVG'] = aggregated_data.groupby('ATM ID')['OSSZEG'].rolling(window=42, min_periods=1).mean().reset_index(0, drop=True)
aggregated_data['WEEKLY_AVG_2_WEEKS_AGO'] = aggregated_data.groupby('ATM ID')['WEEKLY_AVG'].shift(84)

aggregated_data = aggregated_data.dropna()

print(f"Végleges adathalmaz: {aggregated_data.shape}")
print(f"ATM-ek száma: {aggregated_data['ATM ID'].nunique()}")

# ── 4. Feature konfiguráció és előrejelzési horizontok ───────────────────────

all_features = [
    'OSSZEG', 'HOUR', 'WEEKDAY', 'IS_WEEKEND', 'DAY_OF_MONTH', 'IS_HOLIDAY', 'IS_MONTH_END',
    'LAG_126', 'LAG_43', 'LAG_47', 'ATM_AVG', 'DAILY_AVG_1_WEEK_AGO', 'WEEKLY_AVG_2_WEEKS_AGO'
]

forecast_horizons = [
    {'steps': 1,  'hours': 4,   'name': '4h',     'desc': '4 óra'},
    {'steps': 6,  'hours': 24,  'name': '1day',   'desc': '1 nap'},
    {'steps': 12, 'hours': 48,  'name': '2days',  'desc': '2 nap'},
    {'steps': 42, 'hours': 168, 'name': '1week',  'desc': '1 hét'},
    {'steps': 84, 'hours': 336, 'name': '2weeks', 'desc': '2 hét'},
]

print(f"Feature-ök száma: {len(all_features)}")
print(f"Előrejelzési horizontok: {[h['name'] for h in forecast_horizons]}")

# ── 5. Segédfüggvények ────────────────────────────────────────────────────────

def create_seq2seq_dataset(data, features, look_back=42, forecast_steps=1):
    all_X, all_Y = [], []
    for atm_id in data['ATM ID'].unique():
        atm_data = data[data['ATM ID'] == atm_id].sort_values('DATUM_IDO')
        if len(atm_data) < look_back + forecast_steps + 50:
            continue
        feature_values = atm_data[features].values
        osszeg_values = atm_data['OSSZEG'].values
        X_atm, Y_atm = [], []
        for i in range(len(feature_values) - look_back - forecast_steps + 1):
            X_atm.append(feature_values[i:(i + look_back)])
            Y_atm.append(osszeg_values[(i + look_back):(i + look_back + forecast_steps)])
        if len(X_atm) > 0:
            all_X.extend(X_atm)
            all_Y.extend(Y_atm)
    return np.array(all_X), np.array(all_Y)


def temporal_split_by_date(data, train_ratio=0.7, val_ratio=0.15):
    all_dates = sorted(data['DATUM_IDO'].unique())
    train_end_idx = int(len(all_dates) * train_ratio)
    val_end_idx = int(len(all_dates) * (train_ratio + val_ratio))
    train_cutoff = all_dates[train_end_idx]
    val_cutoff = all_dates[val_end_idx]
    return train_cutoff, val_cutoff

# ── 6. Adatfelosztás ──────────────────────────────────────────────────────────

train_cutoff, val_cutoff = temporal_split_by_date(aggregated_data)

train_data = aggregated_data[aggregated_data['DATUM_IDO'] <= train_cutoff]
val_data   = aggregated_data[(aggregated_data['DATUM_IDO'] > train_cutoff) &
                              (aggregated_data['DATUM_IDO'] <= val_cutoff)]
test_data  = aggregated_data[aggregated_data['DATUM_IDO'] > val_cutoff]

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} rekord")
print(f"Train vége: {train_cutoff}, Val vége: {val_cutoff}")

# ── 7. Modell architektúra ────────────────────────────────────────────────────

def build_seq2seq_model(arch_type, n_neurons, dropout_rate, look_back, n_features, forecast_steps):
    inputs = Input(shape=(look_back, n_features), name='encoder_input')

    if arch_type == 'LSTM':
        encoded = LSTM(n_neurons, return_sequences=False, name='encoder_lstm')(inputs)
    elif arch_type == 'GRU':
        encoded = GRU(n_neurons, return_sequences=False, name='encoder_gru')(inputs)

    encoded = Dropout(dropout_rate, name='encoder_dropout')(encoded)
    decoded = RepeatVector(forecast_steps, name='repeat_vector')(encoded)

    if arch_type == 'LSTM':
        decoded = LSTM(n_neurons, return_sequences=True, name='decoder_lstm')(decoded)
    elif arch_type == 'GRU':
        decoded = GRU(n_neurons, return_sequences=True, name='decoder_gru')(decoded)

    decoded = Dropout(dropout_rate, name='decoder_dropout')(decoded)
    outputs = TimeDistributed(Dense(1), name='output_layer')(decoded)

    model = Model(inputs=inputs, outputs=outputs, name=f'{arch_type}_seq2seq')
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
    return model

# ── 8. Tanítás és kiértékelés ─────────────────────────────────────────────────

architectures = [
    {'type': 'LSTM', 'neurons': 128, 'dropout': 0.2},
    {'type': 'GRU',  'neurons': 128, 'dropout': 0.2},
]

nmae_denominator = 9994121
all_results = []

os.makedirs('models', exist_ok=True)

for arch in architectures:
    arch_type    = arch['type']
    n_neurons    = arch['neurons']
    dropout_rate = arch['dropout']

    print(f"\n{'='*60}")
    print(f"Architektúra: {arch_type} ({n_neurons} neuron, dropout={dropout_rate})")
    print(f"{'='*60}")

    for horizon in forecast_horizons:
        forecast_steps = horizon['steps']
        forecast_name  = horizon['name']

        print(f"\n  Horizont: {forecast_name} ({forecast_steps} lépés = {horizon['hours']}h)")

        X_train, Y_train = create_seq2seq_dataset(train_data, all_features, look_back, forecast_steps)
        X_test,  Y_test  = create_seq2seq_dataset(test_data,  all_features, look_back, forecast_steps)

        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()

        X_train_reshaped = X_train.reshape(-1, len(all_features))
        scaler_X.fit(X_train_reshaped)
        scaler_Y.fit(Y_train.reshape(-1, 1))

        X_train_scaled = scaler_X.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled  = scaler_X.transform(X_test.reshape(-1, len(all_features))).reshape(X_test.shape)
        Y_train_scaled = scaler_Y.transform(Y_train.reshape(-1, 1)).reshape(Y_train.shape)

        model = build_seq2seq_model(arch_type, n_neurons, dropout_rate, look_back, len(all_features), forecast_steps)

        checkpoint_path = f'models/multihorizon_{arch_type}_{forecast_name}.keras'
        callbacks = [
            EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
            ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, verbose=0),
        ]

        history = model.fit(
            X_train_scaled, Y_train_scaled,
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        predicted_scaled  = model.predict(X_test_scaled, verbose=0)
        predicted_original = scaler_Y.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(predicted_scaled.shape)
        y_test_original    = Y_test

        predicted_flat = predicted_original.flatten()
        actual_flat    = y_test_original.flatten()

        mae          = mean_absolute_error(actual_flat, predicted_flat)
        r2           = r2_score(actual_flat, predicted_flat)
        nmae         = mae / nmae_denominator
        actual_epochs = len(history.history['loss'])

        print(f"    MAE: {mae/1000:.1f}K HUF, R²: {r2:.3f}, NMAE: {nmae:.3f}, Epochs: {actual_epochs}")

        all_results.append({
            'Architecture':    arch_type,
            'Forecast_Horizon': forecast_name,
            'Forecast_Steps':  forecast_steps,
            'Forecast_Hours':  horizon['hours'],
            'N_Features':      len(all_features),
            'MAE':             round(mae, 1),
            'R2':              round(r2, 3),
            'NMAE':            round(nmae, 3),
            'Epochs':          actual_epochs,
        })

# ── 9. Eredmények összesítése ─────────────────────────────────────────────────

results_df = pd.DataFrame(all_results)

print("\nMulti-horizon előrejelzési eredmények")
print(f"NMAE normalizáció: átlag = {nmae_denominator/1000:.0f}K HUF")
print("=" * 60)

for arch_type in ['LSTM', 'GRU']:
    print(f"\n{arch_type} eredmények:")
    arch_results = results_df[results_df['Architecture'] == arch_type].sort_values('Forecast_Steps')
    print(arch_results[['Forecast_Horizon', 'Forecast_Steps', 'Forecast_Hours', 'MAE', 'NMAE', 'R2']].to_string(index=False))

results_df.to_csv('results.csv', index=False)
print("\nEredmények mentve: results.csv")

overall_best = results_df.loc[results_df['R2'].idxmax()]
print(f"\nLegjobb konfiguráció:")
print(f"  Architektúra: {overall_best['Architecture']}")
print(f"  Horizont: {overall_best['Forecast_Horizon']} ({overall_best['Forecast_Steps']} lépés = {overall_best['Forecast_Hours']}h)")
print(f"  MAE: {overall_best['MAE']/1000:.1f}K HUF, R²: {overall_best['R2']}, NMAE: {overall_best['NMAE']}")
