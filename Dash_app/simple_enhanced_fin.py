import warnings
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore", category=DeprecationWarning)
px.defaults.template = "plotly_white"

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap"
    ],
    suppress_callback_exceptions=True
)
server = app.server

# Create sample data for demonstration
df = pd.DataFrame({
    'Country name': ['Rwanda', 'Kenya', 'Uganda', 'Tanzania', 'Burundi'] * 8,
    'Year': [2011, 2014, 2017, 2021] * 10,
    'Indicator': ['Account (% age 15+)', 'Mobile money account (% age 15+)', 'Made or received a digital payment (% age 15+)'] * 13 + ['Account (% age 15+)'],
    'Indicator value': [32.76, 45.2, 67.8, 89.1, 25.3, 38.9, 52.1, 71.4, 18.5, 29.7, 41.2, 58.9, 22.1, 35.4, 48.7, 65.3, 15.8, 28.2, 39.6, 54.7] * 2,
    'Region': ['Sub-Saharan Africa'] * 40,
    'Income group': ['Lower middle income'] * 40
})

# Define inline styles
styles = {
    'navbar': {
        'background': 'linear-gradient(135deg, #144d14 0%, #276749 100%)',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.15)',
        'borderBottom': '3px solid #1b4d20'
    },
    'container': {
        'fontFamily': 'Lato, sans-serif',
        'backgroundColor': '#f8f9fa'
    },
    'section_title': {
        'color': '#144d14',
        'fontWeight': '700',
        'fontSize': '2rem',
        'marginBottom': '1.5rem',
        'textAlign': 'center',
        'borderBottom': '3px solid #e8f5e8',
        'paddingBottom': '1rem'
    },
    'info_card': {
        'background': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'padding': '1.5rem',
        'marginBottom': '1.5rem',
        'borderLeft': '4px solid #144d14'
    },
    'doc_section': {
        'background': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '2rem',
        'overflow': 'hidden'
    },
    'doc_header': {
        'background': 'linear-gradient(135deg, #144d14 0%, #276749 100%)',
        'color': 'white',
        'padding': '1.5rem',
        'fontWeight': '600',
        'fontSize': '1.3rem'
    },
    'doc_content': {
        'padding': '2rem'
    },
    'stat_item': {
        'background': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'padding': '1.5rem',
        'textAlign': 'center',
        'margin': '0.5rem',
        'borderTop': '4px solid #144d14'
    }
}

# --- Documentation Content ---
def create_documentation_content():
    return html.Div(style={'padding': '2rem'}, children=[
        html.H1("Financial Inclusion & Digital Payments Documentation", 
                style={'textAlign': 'center', 'color': '#144d14', 'marginBottom': '2rem'}),
        
        # Why Financial Inclusion Matters
        html.Div(style=styles['doc_section'], children=[
            html.Div("Why Financial Inclusion Matters", style=styles['doc_header']),
            html.Div(style=styles['doc_content'], children=[
                html.Div(style={'background': '#e8f5e8', 'padding': '1rem', 'borderRadius': '8px', 'marginBottom': '1rem'}, children=[
                    html.H4("Poverty Reduction", style={'color': '#144d14', 'marginBottom': '0.5rem'}),
                    html.P("Financial access boosts economic growth by enabling individuals and businesses to save, invest, and manage risks more effectively.")
                ]),
                html.Div(style={'background': '#e8f5e8', 'padding': '1rem', 'borderRadius': '8px', 'marginBottom': '1rem'}, children=[
                    html.H4("Empowerment", style={'color': '#144d14', 'marginBottom': '0.5rem'}),
                    html.P("Digital tools enhance service access by providing convenient, affordable, and secure ways to conduct financial transactions.")
                ])
            ])
        ]),
        
        # Global Progress
        html.Div(style=styles['doc_section'], children=[
            html.Div("Global Account Ownership Progress", style=styles['doc_header']),
            html.Div(style=styles['doc_content'], children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'margin': '2rem 0'}, children=[
                    html.Div(style=styles['stat_item'], children=[
                        html.Span("46.83%", style={'fontSize': '2rem', 'fontWeight': '700', 'color': '#144d14', 'display': 'block'}),
                        html.Span("Global 2011", style={'fontSize': '0.9rem', 'color': '#6c757d', 'marginTop': '0.5rem'})
                    ]),
                    html.Div(style=styles['stat_item'], children=[
                        html.Span("70.32%", style={'fontSize': '2rem', 'fontWeight': '700', 'color': '#144d14', 'display': 'block'}),
                        html.Span("Global 2021", style={'fontSize': '0.9rem', 'color': '#6c757d', 'marginTop': '0.5rem'})
                    ]),
                    html.Div(style=styles['stat_item'], children=[
                        html.Span("32.76%", style={'fontSize': '2rem', 'fontWeight': '700', 'color': '#144d14', 'display': 'block'}),
                        html.Span("Rwanda 2011", style={'fontSize': '0.9rem', 'color': '#6c757d', 'marginTop': '0.5rem'})
                    ])
                ])
            ])
        ]),
        
        # Key Insights
        html.Div(style=styles['doc_section'], children=[
            html.Div("Key Insights & Recommendations", style=styles['doc_header']),
            html.Div(style=styles['doc_content'], children=[
                dbc.Row([
                    dbc.Col([
                        html.Div(style=styles['info_card'], children=[
                            html.H3("Demographic Impact", style={'color': '#144d14'}),
                            html.P("Income, education, gender, and age profoundly influence financial and digital service adoption.")
                        ])
                    ], md=4),
                    dbc.Col([
                        html.Div(style=styles['info_card'], children=[
                            html.H3("Digital Competency", style={'color': '#144d14'}),
                            html.P("Digital literacy and access to infrastructure are critical enablers for financial inclusion.")
                        ])
                    ], md=4),
                    dbc.Col([
                        html.Div(style=styles['info_card'], children=[
                            html.H3("Targeted Strategies", style={'color': '#144d14'}),
                            html.P("Specific strategies needed to include underserved groups: the poor, women, and youth.")
                        ])
                    ], md=4)
                ])
            ])
        ])
    ])

# --- Layout ---
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand='Financial Inclusion and Digital Payment Adoption Dashboard',
        color='primary',
        dark=True,
        className="mb-4",
        style=styles['navbar']
    ),
    dbc.Row([
        dbc.Col([
            dcc.Tabs(
                id="tabs",
                value='tab-home',
                children=[
                    dcc.Tab(label='🏠 Home', value='tab-home'),
                    dcc.Tab(label='📈 Trends', value='tab-trends'),
                    dcc.Tab(label='👥 Demographics', value='tab-demographics'),
                    dcc.Tab(label='📚 Documentation', value='tab-documentation'),
                ],
                vertical=True
            )
        ], width=2),
        dbc.Col(html.Div(id="tabs-content", style={'padding': '16px'}), width=10)
    ])
], fluid=True, style=styles['container'])

# --- Tab Content Callback ---
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-home':
        return html.Div(style={'padding': '2rem'}, children=[
            html.H1("Welcome to the Global Financial Inclusion Dashboard", 
                   style={'color': '#144d14', 'textAlign': 'center', 'marginBottom': '2rem'}),
            html.P("Gain powerful insights into how adults around the world access, manage, and use financial services from bank accounts to mobile money and digital payments.",
                  style={'fontSize': '1.1rem', 'textAlign': 'center', 'marginBottom': '2rem'}),
            
            dbc.Row([
                dbc.Col([
                    html.Div(style=styles['info_card'], children=[
                        html.H3("🌍 Global Reach", style={'color': '#144d14'}),
                        html.P("Explore financial inclusion data from countries worldwide, with special focus on Sub-Saharan Africa and Rwanda's remarkable progress.")
                    ])
                ], md=6),
                dbc.Col([
                    html.Div(style=styles['info_card'], children=[
                        html.H3("📱 Digital Innovation", style={'color': '#144d14'}),
                        html.P("Track the rise of mobile money and digital payments as key drivers of financial inclusion.")
                    ])
                ], md=6)
            ]),
            
            html.H2("What This Dashboard Offers", style={'color': '#276749', 'marginTop': '2rem'}),
            html.Ul([
                html.Li("Explore global and regional trends in financial account ownership"),
                html.Li("Analyze digital payment adoption across countries"),
                html.Li("Access comprehensive documentation and research insights")
            ], style={'fontSize': '1.05rem'})
        ])

    elif tab == 'tab-trends':
        return html.Div([
            html.H4("📈 Trends in Financial Inclusion", style=styles['section_title']),
            html.P("Explore how financial inclusion indicators have evolved over time.", 
                  style={'textAlign': 'center', 'marginBottom': '2rem'}),
            dcc.Graph(
                figure=px.line(
                    df[df['Indicator'] == 'Account (% age 15+)'].groupby(['Country name', 'Year'])['Indicator value'].mean().reset_index(),
                    x='Year', y='Indicator value', color='Country name',
                    title='Account Ownership Trends by Country'
                ).update_layout(
                    title_font=dict(size=20, family='Lato, sans-serif', color='#144d14'),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
            )
        ])
        
    elif tab == 'tab-demographics':
        return html.Div([
            html.H4("👥 Demographics Analysis", style=styles['section_title']),
            html.P("Understand how demographic factors influence financial service adoption.", 
                  style={'textAlign': 'center', 'marginBottom': '2rem'}),
            dcc.Graph(
                figure=px.bar(
                    df.groupby('Country name')['Indicator value'].mean().reset_index().sort_values('Indicator value'),
                    x='Indicator value', y='Country name', orientation='h',
                    title='Average Financial Inclusion by Country',
                    color_discrete_sequence=['#276749']
                ).update_layout(
                    title_font=dict(size=20, family='Lato, sans-serif', color='#144d14'),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
            )
        ])
        
    elif tab == 'tab-documentation':
        return create_documentation_content()

if __name__ == '__main__':
    app.run(debug=False,port=8058)

