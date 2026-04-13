import dash
from dash import dcc, html, Input, Output, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
import numpy as np

# Load data & model
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# Calculations
attrition_rate = round(df['Attrition'].value_counts(normalize=True)['Yes'] * 100, 1)
total_emp = len(df)
left = df[df['Attrition'] == 'Yes'].shape[0]

# Feature engineering
df['SalaryBand'] = pd.cut(df['MonthlyIncome'], bins=[0, 3000, 6000, 10000, 20000],
                           labels=['Low', 'Medium', 'High', 'Very High'])
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 60],
                         labels=['18-25', '26-35', '36-45', '46+'])

# At Risk Table
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    if col in encoders:
        df_encoded[col] = encoders[col].transform(df_encoded[col])
probs = model.predict_proba(df_encoded[features])[:, 1]
df['AttritionRisk%'] = (probs * 100).round(1)
at_risk = df[df['Attrition'] == 'No'].nlargest(20, 'AttritionRisk%')[
    ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'YearsAtCompany', 'AttritionRisk%']
].reset_index(drop=True)

departments = ['All'] + list(df['Department'].unique())
CARD = {'backgroundColor': '#1e293b', 'borderRadius': '12px', 'padding': '20px'}

def base(fig):
    fig.update_layout(paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
                      title_font_color='#38bdf8')
    return fig

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "HR Attrition Intelligence Dashboard"

app.layout = html.Div(style={'backgroundColor': '#0f172a', 'minHeight': '100vh',
                              'fontFamily': 'Arial', 'padding': '20px'}, children=[

    # Header
    html.Div(style={'textAlign': 'center', 'marginBottom': '30px'}, children=[
        html.H1("HR Attrition Intelligence Dashboard",
                style={'color': '#38bdf8', 'fontSize': '32px', 'marginBottom': '5px'}),
        html.P("Powered by Random Forest ML | IBM HR Dataset",
               style={'color': '#94a3b8', 'fontSize': '14px'})
    ]),

    # KPI Cards
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px',
                    'marginBottom': '30px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'textAlign': 'center', 'minWidth': '160px'}, children=[
            html.H2(str(total_emp), style={'color': '#38bdf8', 'margin': '0', 'fontSize': '36px'}),
            html.P("Total Employees", style={'color': '#94a3b8', 'margin': '0'})]),
        html.Div(style={**CARD, 'textAlign': 'center', 'minWidth': '160px'}, children=[
            html.H2(str(left), style={'color': '#f87171', 'margin': '0', 'fontSize': '36px'}),
            html.P("Employees Left", style={'color': '#94a3b8', 'margin': '0'})]),
        html.Div(style={**CARD, 'textAlign': 'center', 'minWidth': '160px'}, children=[
            html.H2(f"{attrition_rate}%", style={'color': '#fb923c', 'margin': '0', 'fontSize': '36px'}),
            html.P("Attrition Rate", style={'color': '#94a3b8', 'margin': '0'})]),
        html.Div(style={**CARD, 'textAlign': 'center', 'minWidth': '160px'}, children=[
            html.H2("87.76%", style={'color': '#4ade80', 'margin': '0', 'fontSize': '36px'}),
            html.P("Model Accuracy", style={'color': '#94a3b8', 'margin': '0'})]),
    ]),

    # Department Filter
    html.Div(style={'marginBottom': '20px', 'textAlign': 'center'}, children=[
        html.Label("Filter by Department:", style={'color': '#94a3b8', 'marginRight': '10px', 'fontSize': '16px'}),
        dcc.Dropdown(id='dept-filter',
                     options=[{'label': d, 'value': d} for d in departments],
                     value='All', clearable=False,
                     style={'width': '300px', 'display': 'inline-block', 'textAlign': 'left'})
    ]),

    # Row 1
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='dept-chart')]),
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='overtime-chart')]),
    ]),

    # Row 2
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='age-chart')]),
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='jobrole-chart')]),
    ]),

    # Row 3
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='salary-chart')]),
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='satisfaction-chart')]),
    ]),

    # Row 4
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='gender-chart')]),
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='wlb-chart')]),
    ]),

    # Row 5
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='years-chart')]),
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='distance-chart')]),
    ]),

    # Row 6
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='education-chart')]),
        html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[dcc.Graph(id='agegroup-chart')]),
    ]),

    # Heatmap
    html.Div(style={**CARD, 'marginBottom': '20px'}, children=[dcc.Graph(id='heatmap-chart')]),

    # Feature Importance
    html.Div(style={**CARD, 'marginBottom': '20px'}, children=[
        dcc.Graph(figure=base(px.bar(
            x=[0.083, 0.081, 0.060, 0.049, 0.048, 0.046, 0.046, 0.046, 0.044, 0.039],
            y=['OverTime', 'MonthlyIncome', 'Age', 'DailyRate', 'YearsAtCompany',
               'StockOptionLevel', 'MonthlyRate', 'TotalWorkingYears', 'HourlyRate', 'DistanceFromHome'],
            orientation='h', title='Top 10 Feature Importances (Random Forest)',
            template='plotly_dark', color_discrete_sequence=['#38bdf8']
        ).update_layout(paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
                        title_font_color='#38bdf8',
                        yaxis={'categoryorder': 'total ascending'})))
    ]),

    # Live Prediction Form
    html.Div(style={**CARD, 'marginBottom': '20px'}, children=[
        html.H3("🤖 Live Attrition Risk Predictor",
                style={'color': '#38bdf8', 'marginBottom': '20px'}),
        html.P("Enter employee details to predict attrition risk:",
               style={'color': '#94a3b8', 'marginBottom': '20px'}),
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'marginBottom': '20px'}, children=[
            html.Div(children=[
                html.Label("Age", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Input(id='p-age', type='number', value=30, min=18, max=60,
                          style={'width': '120px', 'padding': '8px', 'borderRadius': '6px',
                                 'backgroundColor': '#0f172a', 'color': 'white',
                                 'border': '1px solid #334155', 'display': 'block'})]),
            html.Div(children=[
                html.Label("Monthly Income ($)", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Input(id='p-income', type='number', value=5000, min=1000, max=20000,
                          style={'width': '150px', 'padding': '8px', 'borderRadius': '6px',
                                 'backgroundColor': '#0f172a', 'color': 'white',
                                 'border': '1px solid #334155', 'display': 'block'})]),
            html.Div(children=[
                html.Label("OverTime", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Dropdown(id='p-overtime',
                             options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}],
                             value='No', clearable=False,
                             style={'width': '120px'})]),
            html.Div(children=[
                html.Label("Department", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Dropdown(id='p-dept',
                             options=[{'label': d, 'value': d} for d in df['Department'].unique()],
                             value='Sales', clearable=False,
                             style={'width': '200px'})]),
            html.Div(children=[
                html.Label("Job Role", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Dropdown(id='p-jobrole',
                             options=[{'label': j, 'value': j} for j in df['JobRole'].unique()],
                             value='Sales Executive', clearable=False,
                             style={'width': '200px'})]),
            html.Div(children=[
                html.Label("Years at Company", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Input(id='p-years', type='number', value=3, min=0, max=40,
                          style={'width': '120px', 'padding': '8px', 'borderRadius': '6px',
                                 'backgroundColor': '#0f172a', 'color': 'white',
                                 'border': '1px solid #334155', 'display': 'block'})]),
            html.Div(children=[
                html.Label("Job Satisfaction (1-4)", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Input(id='p-satisfaction', type='number', value=3, min=1, max=4,
                          style={'width': '120px', 'padding': '8px', 'borderRadius': '6px',
                                 'backgroundColor': '#0f172a', 'color': 'white',
                                 'border': '1px solid #334155', 'display': 'block'})]),
            html.Div(children=[
                html.Label("Work Life Balance (1-4)", style={'color': '#94a3b8', 'fontSize': '13px'}),
                dcc.Input(id='p-wlb', type='number', value=3, min=1, max=4,
                          style={'width': '150px', 'padding': '8px', 'borderRadius': '6px',
                                 'backgroundColor': '#0f172a', 'color': 'white',
                                 'border': '1px solid #334155', 'display': 'block'})]),
        ]),
        html.Button("🔍 Predict Attrition Risk", id='predict-btn',
                    style={'backgroundColor': '#38bdf8', 'color': '#0f172a', 'border': 'none',
                           'padding': '12px 24px', 'borderRadius': '8px', 'fontSize': '15px',
                           'cursor': 'pointer', 'fontWeight': 'bold'}),
        html.Div(id='prediction-output', style={'marginTop': '20px'})
    ]),

    # At Risk Table + Download
    html.Div(style={**CARD, 'marginBottom': '20px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                        'alignItems': 'center', 'marginBottom': '15px'}, children=[
            html.H3("🚨 Top 20 At-Risk Employees", style={'color': '#f87171', 'margin': '0'}),
            html.Button("📥 Download Report", id='download-btn',
                        style={'backgroundColor': '#4ade80', 'color': '#0f172a', 'border': 'none',
                               'padding': '10px 20px', 'borderRadius': '8px',
                               'cursor': 'pointer', 'fontWeight': 'bold'}),
        ]),
        dcc.Download(id='download-report'),
        html.P("Currently employed staff with highest predicted attrition probability",
               style={'color': '#94a3b8', 'fontSize': '13px', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=at_risk.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in at_risk.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'backgroundColor': '#0f172a', 'color': '#e2e8f0',
                        'border': '1px solid #1e293b', 'padding': '10px',
                        'fontFamily': 'Arial', 'fontSize': '13px'},
            style_header={'backgroundColor': '#1e293b', 'color': '#38bdf8',
                          'fontWeight': 'bold', 'border': '1px solid #334155'},
            style_data_conditional=[{
                'if': {'filter_query': '{AttritionRisk%} > 50'},
                'backgroundColor': '#450a0a', 'color': '#fca5a5'
            }],
            page_size=10
        )
    ]),

    # Footer
    html.Div(style={'textAlign': 'center', 'color': '#475569', 'fontSize': '13px', 'marginTop': '20px'}, children=[
        html.P("Built by Chatrapathi Reddy | Random Forest Classifier | IBM HR Analytics Dataset")
    ])
])


# ── CHARTS CALLBACK ──
@app.callback(
    [Output('dept-chart', 'figure'),
     Output('overtime-chart', 'figure'),
     Output('age-chart', 'figure'),
     Output('jobrole-chart', 'figure'),
     Output('salary-chart', 'figure'),
     Output('satisfaction-chart', 'figure'),
     Output('gender-chart', 'figure'),
     Output('wlb-chart', 'figure'),
     Output('years-chart', 'figure'),
     Output('distance-chart', 'figure'),
     Output('education-chart', 'figure'),
     Output('agegroup-chart', 'figure'),
     Output('heatmap-chart', 'figure')],
    Input('dept-filter', 'value')
)
def update_charts(selected_dept):
    filtered = df if selected_dept == 'All' else df[df['Department'] == selected_dept]

    # 1. Dept
    dept_data = filtered.groupby('Department')['Attrition'].apply(
        lambda x: (x == 'Yes').sum()).reset_index()
    dept_fig = base(px.bar(dept_data, x='Department', y='Attrition',
                           title='Attrition by Department', template='plotly_dark',
                           color='Attrition', color_continuous_scale='reds'))

    # 2. OverTime
    ot_fig = base(px.pie(filtered, names='OverTime', title='OverTime Distribution',
                         template='plotly_dark', color='OverTime',
                         color_discrete_map={'Yes': '#f87171', 'No': '#4ade80'}))

    # 3. Age
    age_fig = base(px.histogram(filtered, x='Age', color='Attrition',
                                title='Age vs Attrition', template='plotly_dark',
                                barmode='overlay',
                                color_discrete_map={'Yes': '#f87171', 'No': '#38bdf8'}))

    # 4. Job Role
    role_data = filtered.groupby('JobRole')['Attrition'].apply(
        lambda x: (x == 'Yes').sum()).reset_index()
    role_fig = base(px.bar(role_data, x='Attrition', y='JobRole', orientation='h',
                           title='Attrition by Job Role', template='plotly_dark',
                           color='Attrition', color_continuous_scale='reds'))
    role_fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    # 5. Salary Band
    sal_data = filtered.groupby('SalaryBand', observed=False)['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    sal_data.columns = ['SalaryBand', 'AttritionRate']
    sal_fig = base(px.bar(sal_data, x='SalaryBand', y='AttritionRate',
                          title='Attrition Rate by Salary Band (%)', template='plotly_dark',
                          color='AttritionRate', color_continuous_scale='reds'))

    # 6. Job Satisfaction
    sat_data = filtered.groupby('JobSatisfaction')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    sat_data.columns = ['JobSatisfaction', 'AttritionRate']
    sat_data['JobSatisfaction'] = sat_data['JobSatisfaction'].map(
        {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
    sat_fig = base(px.bar(sat_data, x='JobSatisfaction', y='AttritionRate',
                          title='Attrition Rate by Job Satisfaction (%)', template='plotly_dark',
                          color='AttritionRate', color_continuous_scale='reds'))

    # 7. Gender
    gen_fig = base(px.pie(filtered, names='Gender', color='Gender',
                          title='Attrition by Gender', template='plotly_dark',
                          color_discrete_map={'Male': '#38bdf8', 'Female': '#f472b6'}))

    # 8. Work Life Balance
    wlb_data = filtered.groupby('WorkLifeBalance')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    wlb_data.columns = ['WorkLifeBalance', 'AttritionRate']
    wlb_data['WorkLifeBalance'] = wlb_data['WorkLifeBalance'].map(
        {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'})
    wlb_fig = base(px.bar(wlb_data, x='WorkLifeBalance', y='AttritionRate',
                          title='Attrition Rate by Work Life Balance (%)', template='plotly_dark',
                          color='AttritionRate', color_continuous_scale='reds'))

    # 9. Years at Company
    years_data = filtered.groupby('YearsAtCompany')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    years_data.columns = ['YearsAtCompany', 'AttritionRate']
    years_fig = base(px.line(years_data, x='YearsAtCompany', y='AttritionRate',
                             title='Attrition Rate by Years at Company (%)',
                             template='plotly_dark', markers=True,
                             color_discrete_sequence=['#38bdf8']))

    # 10. Distance from Home
    dist_data = filtered.groupby(pd.cut(filtered['DistanceFromHome'],
                                         bins=[0, 5, 10, 15, 20, 29]),
                                  observed=False)['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    dist_data.columns = ['DistanceRange', 'AttritionRate']
    dist_data['DistanceRange'] = dist_data['DistanceRange'].astype(str)
    dist_fig = base(px.bar(dist_data, x='DistanceRange', y='AttritionRate',
                           title='Attrition Rate by Distance from Home (%)',
                           template='plotly_dark', color='AttritionRate',
                           color_continuous_scale='reds'))

    # 11. Education Field
    edu_data = filtered.groupby('EducationField')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    edu_data.columns = ['EducationField', 'AttritionRate']
    edu_fig = base(px.bar(edu_data, x='AttritionRate', y='EducationField', orientation='h',
                          title='Attrition Rate by Education Field (%)', template='plotly_dark',
                          color='AttritionRate', color_continuous_scale='reds'))
    edu_fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    # 12. Age Group
    ag_data = filtered.groupby('AgeGroup', observed=False)['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    ag_data.columns = ['AgeGroup', 'AttritionRate']
    ag_fig = base(px.bar(ag_data, x='AgeGroup', y='AttritionRate',
                         title='Attrition Rate by Age Group (%)', template='plotly_dark',
                         color='AttritionRate', color_continuous_scale='reds'))

    # 13. Heatmap
    heatmap_data = filtered.groupby(['Department', 'JobRole'])['Attrition'].apply(
        lambda x: round((x == 'Yes').mean() * 100, 1)).reset_index()
    heatmap_data.columns = ['Department', 'JobRole', 'AttritionRate']
    heatmap_pivot = heatmap_data.pivot(
        index='JobRole', columns='Department', values='AttritionRate').fillna(0)
    heatmap_fig = base(go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale='Reds', text=heatmap_pivot.values,
        texttemplate='%{text}%'
    )))
    heatmap_fig.update_layout(title='Attrition Heatmap: Department vs Job Role',
                               title_font_color='#38bdf8', template='plotly_dark',
                               paper_bgcolor='#1e293b', plot_bgcolor='#1e293b')

    return (dept_fig, ot_fig, age_fig, role_fig, sal_fig, sat_fig,
            gen_fig, wlb_fig, years_fig, dist_fig, edu_fig, ag_fig, heatmap_fig)


# ── PREDICTION CALLBACK ──
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State('p-age', 'value'), State('p-income', 'value'),
     State('p-overtime', 'value'), State('p-dept', 'value'),
     State('p-jobrole', 'value'), State('p-years', 'value'),
     State('p-satisfaction', 'value'), State('p-wlb', 'value')],
    prevent_initial_call=True
)
def predict_attrition(n_clicks, age, income, overtime, dept, jobrole, years, satisfaction, wlb):
    try:
        row = {}
        for col in features:
            if col in df.select_dtypes(include='number').columns:
                row[col] = float(df[col].median())
            else:
                row[col] = 0.0

        row['Age'] = float(age or 30)
        row['MonthlyIncome'] = float(income or 5000)
        row['YearsAtCompany'] = float(years or 3)
        row['JobSatisfaction'] = float(satisfaction or 3)
        row['WorkLifeBalance'] = float(wlb or 3)

        for col, val in [('OverTime', overtime), ('Department', dept), ('JobRole', jobrole)]:
            if col in encoders and val is not None:
                try:
                    row[col] = float(encoders[col].transform([val])[0])
                except:
                    row[col] = 0.0

        input_df = pd.DataFrame([row])[features]
        prob = model.predict_proba(input_df)[0][1] * 100
        risk_level = "🔴 HIGH RISK" if prob > 60 else "🟡 MEDIUM RISK" if prob > 35 else "🟢 LOW RISK"
        color = '#f87171' if prob > 60 else '#fb923c' if prob > 35 else '#4ade80'

        return html.Div(style={'backgroundColor': '#0f172a', 'borderRadius': '10px',
                               'padding': '20px', 'border': f'2px solid {color}'}, children=[
            html.H3(f"{risk_level}", style={'color': color, 'margin': '0 0 10px 0'}),
            html.H2(f"Attrition Risk: {prob:.1f}%",
                    style={'color': color, 'margin': '0 0 10px 0'}),
            html.P(f"This employee has a {prob:.1f}% probability of leaving the company.",
                   style={'color': '#94a3b8', 'margin': '0'})
        ])
    except Exception as e:
        return html.Div(style={'color': '#f87171', 'padding': '10px'}, children=[
            html.P(f"Error: {str(e)}")
        ])


# ── DOWNLOAD CALLBACK ──
@app.callback(
    Output('download-report', 'data'),
    Input('download-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_report(n_clicks):
    return dcc.send_data_frame(at_risk.to_csv, "at_risk_employees.csv", index=False)


if __name__ == '__main__':
    app.run(debug=True)