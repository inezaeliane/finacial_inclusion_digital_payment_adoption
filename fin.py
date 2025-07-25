import warnings
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server


# Load and prepare data
df = pd.read_excel("world data.xlsx")
df["Indicator value"] = df["Indicator value"].astype(str).str.replace('%', '', regex=False).astype(float)

# --- Trends Page Variables ---
key_indicators = [
    "Account (% age 15+)",
    "Financial institution account (% age 15+)",
    "Mobile money account (% age 15+)",
    "Made or received a digital payment (% age 15+)"  # Fixed indicator name here
]
df_filtered = df[
    (df["Indicator"].isin(key_indicators)) &
    (df["Year"].isin([2011, 2014, 2017, 2021]))
].copy()

# --- Demographics Page Variables ---
dfs_keywords = [
    "mobile money", "digital payment", "internet", "online", "send money",
    "receive money", "used a mobile phone", "used the internet", "e-wallet", "used a mobile"
]
def assign_type(indicator):
    if pd.isna(indicator):
        return "Account"
    indicator_lower = indicator.lower()
    return "DFS" if any(k in indicator_lower for k in dfs_keywords) else "Account"

df["Type"] = df["Indicator"].apply(assign_type)
account_df = df[df["Type"] == "Account"]
dfs_df = df[df["Type"] == "DFS"]

account_data = {
    "Age": account_df[account_df["Indicator"].str.contains("Age", case=False, na=False)],
    "Gender": account_df[account_df["Indicator"].str.contains("female|male", case=False, na=False)],
    "Education": account_df[account_df["Indicator"].str.contains("Education", case=False, na=False)],
    "Income": account_df[account_df["Indicator"].str.contains("Income", case=False, na=False)],
}

dfs_data = {
    "Age": dfs_df[dfs_df["Indicator"].str.contains("Age", case=False, na=False)],
    "Gender": dfs_df[dfs_df["Indicator"].str.contains("female|male", case=False, na=False)],
    "Education": dfs_df[dfs_df["Indicator"].str.contains("Education", case=False, na=False)],
    "Income": dfs_df[dfs_df["Indicator"].str.contains("Income", case=False, na=False)],
}

# --- Barriers Page Variables ---
barrier_keywords = (
    "barrier|reason for not having an account|no account|lack of money|too expensive|too far|"
    "family member has|lack of documentation|lack of trust|religious|no need|not useful|"
    "do not trust financial institutions"
)
df_barriers = df[df["Indicator"].str.contains(barrier_keywords, case=False, regex=True)]

# --- Clustering Function ---
def segment_countries(df, year, region=None):
    df_filtered = df[df['Year'] == year]
    if region:
        df_filtered = df_filtered[df_filtered['Region'] == region]

    df_pivot = df_filtered.pivot_table(index='Country name', columns='Indicator', values='Indicator value')
    df_pivot = df_pivot.dropna(thresh=int(0.5 * len(df_pivot)), axis=1).fillna(df_pivot.mean())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_pivot)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_pivot = df_pivot.assign(PCA1=pca_result[:, 0], PCA2=pca_result[:, 1]).copy()
    df_pivot.reset_index(inplace=True)

    income_info = df_filtered[['Country name', 'Income group']].drop_duplicates()
    df_segmented = df_pivot.merge(income_info, on='Country name', how='left')
    df_segmented.rename(columns={'Income group': 'Segment'}, inplace=True)

    return df_segmented

# --- App Layout ---
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand='Financial Inclusion and Digital Payment Adoption Dashboard',
        color='primary',
        dark=True,
        className="mb-4",
        style={"textAlign": "center", "fontWeight": "bold", "fontSize": "24px"}
    ),
    
    dbc.Row([
        # Left column for vertical tabs
        dbc.Col(
            dcc.Tabs(
                id="tabs",
                value='tab-trends',
                children=[
                    dcc.Tab(label='Home', value='tab-home'),
                    dcc.Tab(label='Trends', value='tab-trends'),
                    dcc.Tab(label='Demographics', value='tab-demographics'),
                    dcc.Tab(label='Barriers', value='tab-barriers'),
                    dcc.Tab(label='Clusters', value='tab-clusters'),
                ],
                vertical=True,
                className='custom-tabs'
            ),
            width=2,
            className="pr-0"
        ),
        
        # Right column for content
        dbc.Col(
            html.Div(id="tabs-content", style={"padding": "16px"}),
            width=10
        )
    ])
], fluid=True)
# --- Render Tabs ---
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    container_style = {
        "backgroundColor": "#e6f7e6",
        "padding": "15px",
        "borderRadius": "12px",
        "boxShadow": "0 4px 8px rgba(50, 150, 50, 0.1)",
        "marginBottom": "30px"
    }

    if tab == 'tab-trends':
        return dbc.Container([
            html.H4("📈 Trends in Financial Inclusion", className="text-center my-3", style={"fontWeight": "bold", "color": "#2d7a2d", "fontSize": "28px"}),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='indicator-dropdown',
                    options=[{"label": ind, "value": ind} for ind in key_indicators],
                    value="Account (% age 15+)", clearable=False
                ), md=6),
                dbc.Col(dcc.RadioItems(
                    id='view-toggle',
                    options=[{"label": v, "value": v} for v in ["Global", "Regional", "Rwanda"]],
                    value="Global",
                    labelStyle={"display": "inline-block", "marginRight": "10px", "color": "#2d7a2d", "fontWeight": "600"}
                ), md=6),
            ], className="mb-3"),
            html.Div(dcc.Graph(id='combined-trend'), style=container_style)
        ], fluid=True)

    elif tab == 'tab-demographics':
        return dbc.Container([
            html.H4("👥 Demographics", className="text-center my-3", style={"fontWeight": "bold", "color": "#2d7a2d", "fontSize": "28px"}),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='data-type',
                    options=[{"label": "Account", "value": "Account"}, {"label": "DFS", "value": "DFS"}],
                    value="Account", clearable=False
                ), md=6),
                dbc.Col(dcc.Dropdown(
                    id='demographic',
                    options=[{"label": f, "value": f} for f in ["Age", "Gender", "Education", "Income"]],
                    value="Age", clearable=False
                ), md=6),
            ], className="mb-3"),
            html.Div(
                dcc.Graph(id='bar-graph'),
                style={**container_style, "height": "600px", "overflowY": "auto"}
            )
        ], fluid=True)

    elif tab == 'tab-barriers':
        return dbc.Container([
            html.H4("🚧 Barriers to Financial Inclusion", className="text-center my-3", style={"fontWeight": "bold", "color": "#2d7a2d", "fontSize": "28px"}),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='region-dropdown',
                    options=[{"label": r, "value": r} for r in sorted(df_barriers["Region"].dropna().unique())],
                    placeholder="Select Region"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id='income-dropdown',
                    options=[{"label": i, "value": i} for i in sorted(df_barriers["Income group"].dropna().unique())],
                    placeholder="Select Income Group"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id='country-dropdown',
                    options=[{"label": c, "value": c} for c in sorted(df_barriers["Country name"].dropna().unique())],
                    placeholder="Select Country"
                ), md=4),
            ], className="mb-3"),
            html.Div(dcc.Graph(id='barrier-graph'), style=container_style)
        ], fluid=True)

    elif tab == 'tab-clusters':
        return dbc.Container([
            html.H4("🗺️ Clusters by Country", className="text-center my-3", style={"fontWeight": "bold", "color": "#2d7a2d", "fontSize": "28px"}),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='year_dropdown',
                    options=[{'label': y, 'value': y} for y in sorted(df['Year'].unique())],
                    value=df['Year'].max()
                ), md=6),
                dbc.Col(dcc.Dropdown(
                    id='region_dropdown',
                    options=[{'label': r, 'value': r} for r in sorted(df['Region'].dropna().unique())],
                    placeholder='All Regions', clearable=True
                ), md=6),
            ], className="mb-3"),
            html.Div(dcc.Graph(id='map_graph'), style=container_style)
        ], fluid=True)

# --- Callbacks ---

@app.callback(
    Output("combined-trend", "figure"),
    Input("indicator-dropdown", "value"),
    Input("view-toggle", "value")
)
def update_combined_trend(indicator, view):
    trend_data = []
    if view == "Global":
        global_df = df_filtered[df_filtered["Indicator"] == indicator].groupby("Year")["Indicator value"].mean().reset_index()
        global_df["Source"] = "Global Average"
        trend_data.append(global_df)

        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)

    elif view == "Regional":
        region_df = df_filtered[df_filtered["Indicator"] == indicator].groupby(["Region", "Year"])["Indicator value"].mean().reset_index()
        region_df.rename(columns={"Region": "Source"}, inplace=True)
        trend_data.append(region_df)

        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)

    elif view == "Rwanda":
        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)

    combined_df = pd.concat(trend_data)

    # Create figure
    fig = px.line(
        combined_df,
        x="Year",
        y="Indicator value",
        color="Source",
        markers=True,
        labels={"Indicator value": "%"}
    )

    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Centered title
    fig.update_layout(
        title={
            'text': f"<b>{indicator} — {view} Comparison</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='darkgreen')
        },
        legend_title_text='Source',
        plot_bgcolor='white',
        font=dict(family="Arial", size=14, color="darkgreen"),
        margin=dict(t=100, b=40, l=40, r=40),
    )

    # Color map
    color_map = {
        "Global Average": "#2ca02c",  # medium green
        "Rwanda": "#1f5e1f",          # dark green
    }

    # Add region-specific colors if Regional
    if view == "Regional":
        region_colors = px.colors.qualitative.Set2
        for i, source in enumerate(combined_df["Source"].unique()):
            if source not in color_map:
                color_map[source] = region_colors[i % len(region_colors)]

    # Apply custom colors
    for trace in fig.data:
        trace.line.color = color_map.get(trace.name, '#66bb66')
        trace.marker.color = color_map.get(trace.name, '#66bb66')

    return fig



@app.callback(
    Output("bar-graph", "figure"),
    Input("data-type", "value"), Input("demographic", "value")
)
def update_demographic_graph(data_type, demographic):
    df_selected = account_data.get(demographic) if data_type == "Account" else dfs_data.get(demographic)
    if df_selected is None or df_selected.empty:
        return px.bar(title="No data available.")
    summary = df_selected.groupby("Indicator")["Indicator value"].mean().reset_index()
    summary_sorted = summary.sort_values(by="Indicator value", ascending=True)

    fig = px.bar(
        summary_sorted,
        x="Indicator value",
        y="Indicator",
        orientation="h",
        labels={"Indicator value": "%", "Indicator": "Indicator"},
        color_discrete_sequence=px.colors.sequential.Greens
    )
    
    # Remove grid lines, bigger height and allow scrolling by setting height + overflow in container (done in layout)
    fig.update_layout(
        title={
            'text': f"<b>{data_type} — {demographic} Overview</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='darkgreen')
        },
        plot_bgcolor='white',
        font=dict(family="Arial", size=14, color="darkgreen"),
        margin=dict(t=100, b=40, l=150, r=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


@app.callback(
    Output("barrier-graph", "figure"),
    Input("region-dropdown", "value"),
    Input("income-dropdown", "value"),
    Input("country-dropdown", "value")
)
def update_barrier_graph(region, income, country):
    filtered = df_barriers.copy()
    if region:
        filtered = filtered[filtered["Region"] == region]
    if income:
        filtered = filtered[filtered["Income group"] == income]
    if country:
        filtered = filtered[filtered["Country name"] == country]
    if filtered.empty:
        return px.bar(title="No data available.")
    summary = filtered.groupby("Indicator")["Indicator value"].mean().reset_index()
    summary_sorted = summary.sort_values(by="Indicator value", ascending=True)  # ascending order as requested

    fig = px.bar(
        summary_sorted,
        x="Indicator value",
        y="Indicator",
        orientation="h",
        labels={"Indicator value": "%", "Indicator": "Barrier"},
        color_discrete_sequence=px.colors.sequential.Greens_r
    )
    fig.update_layout(
        title={
            'text': "<b>Barriers to Financial Inclusion</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='darkgreen')
        },
        plot_bgcolor='white',
        font=dict(family="Arial", size=14, color="darkgreen"),
        margin=dict(t=100, b=40, l=200, r=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


@app.callback(
    Output("map_graph", "figure"),
    Input("year_dropdown", "value"),
    Input("region_dropdown", "value")
)
def update_map_graph(year, region):
    df_segmented = segment_countries(df, year, region)
    fig = px.choropleth(
        df_segmented,
        locations='Country name',
        locationmode='country names',
        color='Segment',
        hover_name='Country name',
        color_continuous_scale=px.colors.sequential.Greens,
        labels={'Segment': 'Income Group'}
    )
    fig.update_layout(
        title={
            'text': "<b>Clusters by Country</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='darkgreen')
        },
        plot_bgcolor='white',
        font=dict(family="Arial", size=14, color="darkgreen"),
        margin=dict(t=100, b=40, l=40, r=40)
    )
    return fig


# Run App
if __name__ == "__main__":
    app.run(debug=True)
