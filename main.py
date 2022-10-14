import io
import re
import base64
import string
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from dash import html, dcc, dash_table, ctx
import statsmodels.formula.api as smf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

mlr_app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

colors = {"graphBackground": "#F5F5F5",
          "background": "#ffffff", "text": "#000000"}

tab_cat_dropdwn = html.Div(id="tab_cat_dropdwn",
                           style={'display': 'block'},
                           children=[
                               html.P("Categories:"),
                               dcc.Dropdown(id='names', options=[
                                   "-----"], value='-----', clearable=False)
                           ])

tab_mlr_dropdwn = html.Div(id="tab_mlr_dropdwn",
                           style={'display': 'block'},
                           children=[
                               html.P("Y:"),
                               dcc.Dropdown(id='regress', options=[
                                   "-----"], value='-----', clearable=False),
                               html.P("X:"),
                               dcc.Dropdown(id='predictors', options=[
                                   "-----"], value='-----', clearable=False, multi=True)
                           ])

fit_model_btn = html.Button('Fit Model', id='fit_model_btn', n_clicks=0)

mlr_app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Div(id="output-data-upload"),
        dcc.Tabs(id="tabs-contents", value='describe', children=[
            dcc.Tab(label='Numerical Summary', value='describe'),
            dcc.Tab(label='Categorical Columns', value='pie_chart'),
            dcc.Tab(label='MLR', value='mlr')]),
        html.Div(id='tabs-content-dropdwn',
                 children=[tab_cat_dropdwn, tab_mlr_dropdwn, html.Hr(), fit_model_btn]),
        html.Div(id='tabs-content-graph')
    ]
)


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith(".csv"):
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        elif filename.endswith(".xls"):
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        elif filename.endswith(".txt"):
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(
                decoded.decode("utf-8")), delimiter=r"\s+")

    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


@mlr_app.callback(
    Output("output-data-upload", "children"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)
def update_table(contents, filename):
    update_tab_cat_opt(contents=contents, filename=filename)
    table = html.Div()
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    page_size=50,  # we have less data in this example, so setting to 20
                    style_table={'height': '150px', 'overflowY': 'auto'}
                ),
                html.Hr()
            ]
        )
    return table


@mlr_app.callback(
    [Output('tab_cat_dropdwn', 'style'),
     Output('tab_mlr_dropdwn', 'style'),
     Output('fit_model_btn', 'style')],
    Input('tabs-contents', 'value'))
def update_style(tab):
    show = {'display': 'block'}
    hide = {'display': 'none'}
    if tab == 'pie_chart':
        return show, hide, hide
    elif tab == 'mlr':
        return hide, show, show
    else:
        return hide, hide, hide


@mlr_app.callback(
    Output('tabs-content-graph', 'children'),
    [Input('tabs-contents', 'value'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('names', 'value'),
        Input('regress', 'value'),
        Input('predictors', 'value'),
        Input('fit_model_btn', 'n_clicks')],
)
def render_content(tab, contents, filename, names, regress, predictors, fit_model_btn):
    fig = {}
    if contents:
        contents = contents[0]
        filename = filename[0]
        raw_data = parse_data(contents, filename)
        if tab == 'pie_chart':
            if names != "-----":
                cross_tab = pd.crosstab(raw_data[names], columns=["count"])
                y = np.array(cross_tab["count"].values)
                mylabels = cross_tab.index.tolist()
                trace = go.Pie(labels=mylabels, values=y, hole=0.3)
                fig = html.Div(
                    [
                        html.H5("Pie Chart Display"),
                        dcc.Graph(
                            figure={
                                "data": [trace]
                            }
                        ),
                        html.Hr()
                    ]
                )
        elif tab == 'describe':
            describe_tab = raw_data.describe()
            describe_tab = describe_tab.reset_index()
            fig = html.Div(
                [
                    html.H5("Describe Table"),
                    dash_table.DataTable(
                        data=describe_tab.to_dict("rows"),
                        columns=[{"name": i, "id": i}
                                 for i in describe_tab.columns],
                        page_size=20,  # we have less data in this example, so setting to 20
                        style_table={
                            'height': '600px', 'overflowY': 'auto'}
                    ),
                    html.Hr()
                ]
            )
        elif tab == "mlr":
            if regress != "-----" and "-----" not in predictors and len(predictors) != 0:
                if "fit_model_btn" == ctx.triggered_id:
                    predictors = ["Q('{}')".format(p) if len(re.findall(
                        '[' ' ' + string.punctuation + '\\r\\t\\n]', p)) > 0 else p for p in predictors]
                    regress = "Q('{}')".format(regress) if len(re.findall(
                        '[' ' ' + string.punctuation + '\\r\\t\\n]', regress)) > 0 else regress
                    model_type = regress+"~"+"+".join(predictors)
                    model = smf.ols(model_type, data=raw_data).fit()
                    res_tables = model.summary()
                    result = res_tables.tables[0].as_html()
                    result = pd.read_html(result, header=0, index_col=0)[0]
                    result = result.reset_index()
                    t_test = res_tables.tables[1].as_html()
                    t_test = pd.read_html(t_test, header=0, index_col=0)[0]
                    t_test = t_test.reset_index()
                    others = res_tables.tables[2].as_html()
                    others = pd.read_html(others, header=0, index_col=0)[0]
                    others = result.reset_index()

                    dash_table1 = dash_table.DataTable(
                        data=result.to_dict("rows"),
                        columns=[{"name": i, "id": i} for i in result.columns],
                        page_size=20,  # we have less data in this example, so setting to 20
                        style_table={'height': '600px', 'overflowY': 'auto'}
                    )
                    dash_table2 = dash_table.DataTable(
                        data=t_test.to_dict("rows"),
                        columns=[{"name": i, "id": i} for i in t_test.columns],
                        page_size=20,  # we have less data in this example, so setting to 20
                        style_table={'height': '600px', 'overflowY': 'auto'}
                    )
                    dash_table3 = dash_table.DataTable(
                        data=others.to_dict("rows"),
                        columns=[{"name": i, "id": i} for i in others.columns],
                        page_size=20,  # we have less data in this example, so setting to 20
                        style_table={'height': '600px', 'overflowY': 'auto'}
                    )

                    fig = html.Div([
                        html.Div([dash_table1], style={
                                 'display': 'inline-block'}),
                        html.Div([dash_table2], style={
                                 'display': 'inline-block'}),
                        html.Div([dash_table3], style={
                                 'display': 'inline-block'})
                    ])

    return fig


@mlr_app.callback(Output('names', 'options'),
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename'))
def update_tab_cat_opt(contents, filename):
    options = ["-----"]
    if contents:
        contents = contents[0]
        filename = filename[0]
        raw_data = parse_data(contents, filename)
        numerical_ = set(raw_data.select_dtypes(
            include=np.number).columns.tolist())
        all_col_n_ = set(raw_data.columns.tolist())
        options = ['-----'] + list(all_col_n_.difference(numerical_))
    return options


@mlr_app.callback(Output('regress', 'options'),
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename'),
                  Input('predictors', 'value'))
def update_reg_opt(contents, filename, pred_val):
    options = ["-----"]
    pred_val = set(pred_val) if isinstance(pred_val, list) else {pred_val}
    if contents:
        contents_ = contents[0]
        filename_ = filename[0]
        raw_data = parse_data(contents_, filename_)
        numerical_ = set(raw_data.select_dtypes(
            include=np.number).columns.tolist())
        options = list(numerical_.difference(pred_val))
    return options


@mlr_app.callback(Output('predictors', 'options'),
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename'),
                  Input('regress', 'value'))
def update_pred_opt(contents, filename, regress_val):
    options = ["-----"]
    regress_val = set(regress_val) if isinstance(
        regress_val, list) else {regress_val}
    if contents:
        contents_ = contents[0]
        filename_ = filename[0]
        raw_data = parse_data(contents_, filename_)
        all_col_n_ = set(raw_data.columns.tolist())
        options = list(all_col_n_.difference(regress_val))
    return options


mlr_app.run_server(mode='jupyterlab', port=8090, dev_tools_ui=False,
                   dev_tools_hot_reload=True, threaded=True)
