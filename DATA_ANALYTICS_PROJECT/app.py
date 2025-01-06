from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted,NotFittedError
import plotly.express as px
import plotly
import plotly.graph_objects as go
import json
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
dataset = None
dataset_prediction = None
dataset_rules = None
model = None
selected_features = None
label_encoders = {}
unique_values = {}
a = 'kkk'
@app.route('/')
def home():
    return render_template('home.html')
        
@app.route('/load_data', methods=['POST'])
def load_data():
    global dataset, dataset_prediction, dataset_rules
    file = request.files['file']

    if not file:
        return redirect(url_for('home'))
    
    dataset = pd.read_csv(file)
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
           dataset[column].fillna(dataset[column].mode()[0]) 
        else:
           dataset[column].fillna(dataset[column].mean()) 

    for column in dataset.columns:
        if len(np.unique(dataset[column])) > 300:
            dataset.drop(column, axis=1, inplace=True)

    dataset_prediction = dataset.copy()
    dataset_rules = dataset.copy()

    return redirect(url_for('choose_option'))



@app.route('/choose_option')
def choose_option():
    return render_template('choose_option.html')


@app.route('/casestudy', methods=['GET', 'POST'])
def casestudy():
    skill_keywords = ["Python", "SQL", "Java", "Excel", "AWS", "Spark"]
    special_r_columns = ['r_yn', "r", "r_language", "r_lang"]
    skill_columns = []
    
    for col in dataset.columns:
        if any(skill.lower() in col.lower() for skill in skill_keywords):
            skill_columns.append(col)

    for col in dataset.columns:
        if col.lower() in special_r_columns:
            skill_columns.append(col)

    skill_columns = list(set(skill_columns)) 
    categorical_features = [col for col in dataset.columns if dataset[col].dtype == 'object' and col not in skill_columns]
    numerical_features = [col for col in dataset.columns if dataset[col].dtype != 'object' and col not in skill_columns]
    
    if request.method == 'POST':
        categorical_features_selected = request.form.getlist('categorical_features')
        numerical_features_selected = request.form.getlist('numerical_features')
        skill_columns_selected = request.form.getlist('skill_columns')
        graph_type = request.form['graph_type']
            
        selected_features = categorical_features_selected + numerical_features_selected + skill_columns_selected

        fig = None

        if graph_type == 'bar':

            if len(categorical_features_selected) == 1 and not numerical_features_selected and not skill_columns_selected:
                categorical_feature = categorical_features_selected[0]
                grouped_data = dataset[categorical_feature].value_counts().reset_index()
                grouped_data.columns = [categorical_feature, 'Count']
                fig = px.bar(grouped_data, x=categorical_feature, y='Count', title=f'Distribution of {categorical_feature}',color_discrete_sequence=['rgba(204, 153, 255, 0.8)'])

            elif len(numerical_features_selected) == 1 and not categorical_features_selected and not skill_columns_selected:
                numerical_feature = numerical_features_selected[0]
                grouped_data = dataset[numerical_feature].value_counts().reset_index()
                grouped_data.columns = [numerical_feature, 'Count']
                fig = px.bar(grouped_data, x=numerical_feature, y='Count', title=f'Distribution of {numerical_feature}',color_discrete_sequence=['rgba(204, 153, 255, 0.8)'])
            
            elif len(skill_columns_selected) == 1 and not categorical_features_selected and not numerical_features_selected:
                skill_column = skill_columns_selected[0]
                df = dataset[[skill_column]]
                df[skill_column] = df[skill_column].apply(lambda x: "yes" if x else "no")
                grouped_data = df[skill_column].value_counts().reset_index()
                grouped_data.columns = [skill_column, 'Count']
                fig = px.bar(grouped_data, x=skill_column, y='Count', title=f'Skill Distribution of {skill_column}',color_discrete_sequence=['rgba(204, 153, 255, 0.8)'])
            
            elif len(skill_columns_selected) == 1 and len(numerical_features_selected) == 1 and not categorical_features_selected:
                skill_column = skill_columns_selected[0]
                numerical_feature = numerical_features_selected[0]
                df = dataset[[skill_column, numerical_feature]]
                df[skill_column] = df[skill_column].apply(lambda x: "yes" if x else "no")
                grouped_data = df.groupby(skill_column)[numerical_feature].mean().reset_index()
                fig = px.bar(
                    grouped_data,
                    x=skill_column, 
                    y=numerical_feature,  
                    labels={skill_column: "Skill", numerical_feature: f"Mean {numerical_feature}"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    title=f'Distribution of {numerical_feature} across skill-{skill_column}',
                    xaxis_title=skill_column,
                    yaxis_title=f'{numerical_feature}'
                )
            
            elif len(categorical_features_selected) == 1 and len(numerical_features_selected) == 1 and not skill_columns_selected:
                categorical_feature = categorical_features_selected[0]
                numerical_feature = numerical_features_selected[0]
                
                grouped_data = dataset.groupby(categorical_feature)[numerical_feature].mean().reset_index()
                grouped_data.sort_values(by=numerical_feature, ascending=False, inplace=True)
                fig = px.bar(
                    grouped_data,
                    x=categorical_feature, 
                    y=numerical_feature,  
                    labels={categorical_feature: "Category", numerical_feature: f"Mean {numerical_feature}"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    title=f'Distribution of {numerical_feature} across {categorical_feature}',
                    xaxis_title=categorical_feature,
                    yaxis_title=f'{numerical_feature}'
                )
            
            elif len(categorical_features_selected) == 1 and len(skill_columns_selected) == 1 and len(numerical_features_selected) == 1:
                numerical_feature = numerical_features_selected[0]
                categorical_feature = categorical_features_selected[0]
                skill_column = skill_columns_selected[0]
                df = dataset[[skill_column,categorical_feature,numerical_feature]]
                df[skill_column] = df[skill_column].apply(lambda x: "yes" if x else "no")
                grouped_data = df.groupby([categorical_feature , skill_column])[numerical_feature].mean().reset_index()
                fig = px.bar(
                    grouped_data,
                    x=categorical_feature,  
                    y=numerical_feature,    
                    color=skill_column,    
                    labels={categorical_feature: "Category", numerical_feature: "Mean Value"}
                )
                fig.update_layout(
                    title=f'Distribution of {numerical_feature} across {categorical_feature} and skill-{skill_column}',
                    xaxis_title=categorical_feature,
                    yaxis_title=f'{numerical_feature}')
                
            elif len(skill_columns_selected) == 2 and not categorical_features_selected and len(numerical_features_selected) == 1:
                numerical_feature = numerical_features_selected[0]
                skill_column_x = skill_columns_selected[0]
                skill_column_y = skill_columns_selected[1]
                df = dataset[[skill_column_x,skill_column_y,numerical_feature]]
                df[skill_column_x] = df[skill_column_x].apply(lambda x: "yes" if x else "no")
                df[skill_column_y] = df[skill_column_y].apply(lambda x: "yes" if x else "no")
                grouped_data = df.groupby([skill_column_x,skill_column_y])[numerical_feature].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=skill_column_x, 
                    y=numerical_feature,    
                    color=skill_column_y,
                    barmode='group',
                    labels={
                                skill_column_x: f"Skill: {skill_column_x}",
                                skill_column_y: f"Skill: {skill_column_y}",
                                numerical_feature: f"Mean {numerical_feature}"
                            }    
                     )
                fig.update_layout(
                        title=f'Distribution of {numerical_feature} across {skill_column_x} and {skill_column_y}',
                        xaxis_title=f'{skill_column_x} (Skill)',
                        yaxis_title=f'Mean {numerical_feature}',
                        legend_title=f'{skill_column_y} (Skill)',
                    )
            
            elif len(categorical_features_selected) == 2 and not skill_columns_selected and len(numerical_features_selected) == 1:
                categorical_group = dataset.groupby(categorical_features_selected)
               
                grouped_data = categorical_group[numerical_features_selected].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=categorical_features_selected[0], 
                    y=numerical_features_selected[0],    
                    color=categorical_features_selected[1] if len(categorical_features_selected) > 1 else None,  
                    labels={categorical_features_selected[0]: "Category", numerical_features_selected[0]: "Mean Value"}
                )
                fig.update_layout(
                    title=f'Distribution of {numerical_features_selected[0]} across {", ".join(categorical_features_selected)}',
                    xaxis_title=", ".join(categorical_features_selected),
                    yaxis_title=f'Mean {numerical_features_selected[0]}'
                )

            elif len(numerical_features_selected) == 2 and not categorical_features_selected and not skill_columns_selected:
                numerical_feature_x = numerical_features_selected[0]
                numerical_feature_y = numerical_features_selected[1]
                
                grouped_data = dataset.groupby(numerical_feature_x)[numerical_feature_y].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=numerical_feature_x,
                    y=numerical_feature_y,
                    labels={numerical_feature_x: "Group", numerical_feature_y: f"Mean of {numerical_feature_y}"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    title=f'Distribution of {numerical_feature_y} grouped by {numerical_feature_x}',
                    xaxis_title=numerical_feature_x,
                    yaxis_title=f'Mean {numerical_feature_y}'
                )
            elif len(numerical_features_selected) == 2 and len(categorical_features_selected) == 1 and not skill_columns_selected:
                numerical_feature_x = numerical_features_selected[0]
                numerical_feature_y = numerical_features_selected[1]
                categorical_feature = categorical_features_selected[0]
                
                grouped_data = dataset.groupby([categorical_feature, numerical_feature_x])[numerical_feature_y].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=numerical_feature_x,
                    y=numerical_feature_y,
                    color=categorical_feature,
                    labels={
                        numerical_feature_x: f"{numerical_feature_x}",
                        numerical_feature_y: f"Mean of {numerical_feature_y}",
                        categorical_feature: f"{categorical_feature}"
                    },
                    barmode="group", 
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    title=f'{numerical_feature_y} grouped by {numerical_feature_x} and {categorical_feature}',
                    xaxis_title=numerical_feature_x,
                    yaxis_title=f'Mean {numerical_feature_y}'
                )
            
            elif len(numerical_features_selected) == 2 and len(skill_columns_selected) == 1 and not categorical_features_selected:
                numerical_feature_x = numerical_features_selected[0]
                numerical_feature_y = numerical_features_selected[1]
                skill_column = skill_columns_selected[0]
                df = dataset[[skill_column, numerical_feature_x, numerical_feature_y]]
                df[skill_column] = df[skill_column].apply(lambda x: "yes" if x else "no")
                
                grouped_data = df.groupby([skill_column, numerical_feature_x])[numerical_feature_y].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=numerical_feature_x,
                    y=numerical_feature_y,
                    color=skill_column,
                    labels={
                        numerical_feature_x: f"{numerical_feature_x}",
                        numerical_feature_y: f"Mean of {numerical_feature_y}",
                        skill_column: f"{skill_column}"
                    },
                    barmode="group",  
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    title=f'{numerical_feature_y} grouped by {numerical_feature_x} and {skill_column}',
                    xaxis_title=numerical_feature_x,
                    yaxis_title=f'Mean {numerical_feature_y}'
                )

            else:
                error_message = (
                    "Please select the correct combination of features for the Bar chart.<br>"
                    "You can choose one of the following options:<br>"
                    "1. A single categorical feature or skill feature or numerical feature.<br>"
                    "2. A combination of a categorical feature and a numerical feature, "
                    "   or a combination of a skill feature and a numerical feature.<br>"
                    "3. A combination of two categorical features with a numerical feature, "
                    "   or a combination of two skill features with a numerical feature.<br>"
                    "Make sure you don't mix multiple categories or skills with incompatible options."
                )

                
                return render_template('casestudy.html', 
                                    error_message=error_message,
                                    categorical_features=categorical_features, 
                                    numerical_features=numerical_features,
                                    skill_columns=skill_columns)

        elif graph_type == 'pie':
            if len(categorical_features_selected) == 1 and not numerical_features_selected and not skill_columns_selected:
                categorical_feature = categorical_features_selected[0]
                grouped_data = dataset[categorical_feature].value_counts().reset_index()
                grouped_data.columns = [categorical_feature, 'Count'] 
                fig = px.pie(grouped_data, names=categorical_feature, values='Count', title=f'Distribution of {categorical_feature}')

            elif len(skill_columns_selected) == 1 and not categorical_features_selected and not numerical_features_selected:
                skill_column = skill_columns_selected[0]
                df = dataset[[skill_column]]
                df[skill_column] = df[skill_column].apply(lambda x: "yes" if x else "no")
                grouped_data = df[skill_column].value_counts().reset_index()
                grouped_data.columns = [skill_column, 'Count']
                fig = px.pie(grouped_data, names=skill_column, values='Count', title=f'Distribution of skill-{skill_column}')

            elif len(skill_columns_selected) == 1 and len(numerical_features_selected) == 1 and not categorical_features_selected:
                skill_column = skill_columns_selected[0]
                numerical_feature = numerical_features_selected[0]
                df = dataset[[skill_column,numerical_feature]]
                df[skill_column] = df[skill_column].apply(lambda x: "yes" if x else "no")
                grouped_data = df.groupby(skill_column)[numerical_feature].mean().reset_index()

                fig = px.pie(
                    grouped_data,
                    names=skill_column, 
                    values=numerical_feature, 
                    title=f'Distribution of {numerical_feature} across {skill_column}'
                )
            
            elif len(categorical_features_selected) == 1 and not skill_columns_selected and len(numerical_features_selected) == 1:
                categorical_feature = categorical_features_selected[0]
                numerical_feature = numerical_features_selected[0]

                grouped_data = dataset.groupby(categorical_feature)[numerical_feature].mean().reset_index()

                fig = px.pie(
                    grouped_data,
                    names=categorical_feature,  
                    values=numerical_feature,  
                    title=f'Distribution of {numerical_feature} across {categorical_feature}'
                )
       
            else:
                error_message = (
                        "Please select the correct combination of features for the pie chart.<br> "
                        "You can choose one of the following options:<br> "
                        "1. A single categorical feature.<br> "
                        "2. A single skill feature.<br>"
                        "3. A skill column along with a numerical feature.<br> "
                        "4. A combination of a categorical feature and a numerical feature.<br> "
                        "Make sure you don't mix multiple categories or skills with incompatible options.")
                
                return render_template('casestudy.html', 
                                    error_message=error_message,
                                    categorical_features=categorical_features, 
                                    numerical_features=numerical_features,
                                    skill_columns=skill_columns)

        elif graph_type == 'box' :
            if  len(selected_features) == 1 and not categorical_features_selected:
                fig = px.box(dataset, y=selected_features[0],color_discrete_sequence=['rgba(204, 153, 255, 0.8)'])
                fig.update_layout(title=f'Box plot of {selected_features[0]}')

            elif len(skill_columns_selected) == 1 and len(numerical_features_selected) == 1 and not categorical_features_selected:
                skill_column = skill_columns_selected[0]
                numerical_feature = numerical_features_selected[0]
                
                fig = px.box(
                    dataset,
                    x=skill_column,
                    y=numerical_feature,
                    title=f'Box Plot of {numerical_feature} by {skill_column}',
                    labels={skill_column: "Skill", numerical_feature: "Value"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    xaxis_title=skill_column,
                    yaxis_title=numerical_feature
                )
            
            elif len(categorical_features_selected) == 1 and len(numerical_features_selected) == 1 and not skill_columns_selected:
                categorical_feature = categorical_features_selected[0]
                numerical_feature = numerical_features_selected[0]
                
                fig = px.box(
                    dataset,
                    x=categorical_feature,
                    y=numerical_feature,
                    title=f'Box Plot of {numerical_feature} by {categorical_feature}',
                    labels={categorical_feature: "Skill", numerical_feature: "Value"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    xaxis_title=categorical_feature,
                    yaxis_title=numerical_feature
                )

            else:
                error_message = (
                        "Please select the correct combination of features for the box plot.<br> "
                        "You can choose one of the following options:<br> "
                        "1. A single numerical feature or skill column.<br> "
                        "2. A combination of a categorical feature and a numerical feature.<br> "
                        "3. A combination of a skill column and a numerical feature.<br> "
                        "Make sure you don't mix multiple categories or skills with incompatible options.")
                
                return render_template('casestudy.html', 
                                    error_message=error_message,
                                    categorical_features=categorical_features, 
                                    numerical_features=numerical_features,
                                    skill_columns=skill_columns)


        elif graph_type == 'histogram' :
            if len(selected_features) == 1 and not categorical_features_selected:
                fig = px.histogram(dataset, x=selected_features[0],color_discrete_sequence=['rgba(204, 153, 255, 0.8)'])
                fig.update_layout(title=f'Histogram of {selected_features[0]}')
            else :
                error_message = (
                        "Please select the correct combination of features for the Histogram .<br> "
                        "You can only select a single numerical feature or skill column.<br> ")
                
                return render_template('casestudy.html', 
                                    error_message=error_message,
                                    categorical_features=categorical_features, 
                                    numerical_features=numerical_features,
                                    skill_columns=skill_columns)

        elif graph_type == 'scatter':
            if len(numerical_features_selected) == 2 and not categorical_features_selected and not skill_columns_selected:

                x_feature = numerical_features_selected[0]
                y_feature = numerical_features_selected[1]
                

                fig = px.scatter(
                    dataset,
                    x=x_feature,
                    y=y_feature,
                    title=f'Scatter Plot of {y_feature} vs {x_feature}',
                    labels={x_feature: f"{x_feature}", y_feature: f"{y_feature}"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    xaxis_title=x_feature,
                    yaxis_title=y_feature
                )

            elif len(numerical_features_selected) == 1 and len(skill_columns_selected) == 1 and not categorical_features_selected:
                x_feature = skill_columns_selected[0]
                y_feature = numerical_features_selected[0]

                fig = px.scatter(
                    dataset,
                    x=x_feature,
                    y=y_feature,
                    title=f'Scatter Plot of {y_feature} vs {x_feature}',
                    labels={x_feature: f"{x_feature}", y_feature: f"{y_feature}"},
                    color_discrete_sequence=['rgba(204, 153, 255, 0.8)']
                )
                fig.update_layout(
                    xaxis_title=x_feature,
                    yaxis_title=y_feature
                )

            else:
                error_message = (
                    "Please select exactly two numerical features for the scatter plot.<br>"
                    "Scatter plots require two numerical features for the x and y axes."
                )
                return render_template(
                    'casestudy.html',
                    error_message=error_message,
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                    skill_columns=skill_columns
                )

        if fig:
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('casestudy.html', graphJSON=graphJSON, 
                                   categorical_features=categorical_features, 
                                   numerical_features=numerical_features,
                                   skill_columns=skill_columns)

    return render_template('casestudy.html', 
                           categorical_features=categorical_features, 
                           numerical_features=numerical_features, 
                           skill_columns=skill_columns)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global model, selected_features, dataset_prediction, label_encoders, unique_values, prediction_result, X_train, X_test, y_train, y_test

    if request.method == 'POST':
        if 'select_target' in request.form:
            target_feature = request.form['target']

            if not target_feature:
                return render_template('prediction.html', 
                                       target_feature=None, 
                                       correlation_table=None, 
                                       features=None, 
                                       trained=False, 
                                       error="Please select a target variable.")

            label_encoders = {}
            dataset_encoded = dataset_prediction.copy()
            unique_values = {}

            for feature in dataset_encoded.columns:
                if dataset_encoded[feature].dtype == 'object':
                    le = LabelEncoder()
                    dataset_encoded[feature] = le.fit_transform(dataset_encoded[feature].astype(str))
                    label_encoders[feature] = le
                    unique_values[feature] = le.classes_.tolist()

            numerical_features = dataset_encoded.select_dtypes(include=np.number).columns.tolist()
            correlation_table = dataset_encoded[numerical_features].corr()[[target_feature]].sort_values(by=target_feature, ascending=False).round(2)
            correlation_table = correlation_table.drop(target_feature)
            
            features = [f for f in numerical_features]

            return render_template('prediction.html', 
                                   target_feature=target_feature, 
                                   correlation_table=correlation_table, 
                                   features=features, 
                                   trained=False)

        elif 'train' in request.form:
            selected_features = request.form.getlist('features')
            target_feature = request.form['target']

            if not selected_features or not target_feature:
                return render_template('prediction.html', 
                                       target_feature=target_feature, 
                                       correlation_table=None, 
                                       features=dataset_prediction.columns, 
                                       trained=False, 
                                       error="Please select features and a target for training.")

            dataset_processed = dataset_prediction.copy()
            for feature in dataset_processed.columns:
                if feature in label_encoders:
                    dataset_processed[feature] = label_encoders[feature].transform(dataset_processed[feature].astype(str))

            X = dataset_processed[selected_features]
            y = dataset_processed[target_feature]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'SVR': SVR(),
                'Random Forest': RandomForestRegressor()
            }

            mse_values = {}
            r2_values = {}

            for model_name, model_instance in models.items():
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_test)
                mse_values[model_name] = mean_squared_error(y_test, y_pred)
                r2_values[model_name] = r2_score(y_test, y_pred)

            mse_graph = go.Figure(data=[go.Bar(x=list(mse_values.keys()), y=list(mse_values.values()),marker_color='rgba(204, 153, 255, 0.8)')])
            mse_graph.update_layout(
                title="Model Comparison - MSE",
                xaxis_title="Model",
                yaxis_title="Mean Squared Error"
            )
            mse_div = mse_graph.to_html(full_html=False)

            r2_graph = go.Figure(data=[go.Bar(x=list(r2_values.keys()), y=list(r2_values.values()),marker_color='rgba(204, 153, 255, 0.8)')])
            r2_graph.update_layout(
                title="Model Comparison - R² Score",
                xaxis_title="Model",
                yaxis_title="R² Score"
            )
            r2_div = r2_graph.to_html(full_html=False)

            return render_template('prediction.html', 
                                mse_graph=mse_div, 
                                r2_graph=r2_div, 
                                target_feature=target_feature, 
                                trained=True, 
                                selected_features=selected_features, 
                                features=dataset_prediction.columns, 
                                unique_values=unique_values)

        elif 'model_selection' in request.form:
            selected_model = request.form['model_selection']

            global selected_model_type
            selected_model_type = selected_model

            return redirect(url_for('prediction_input'))

    numerical_features = dataset_prediction.select_dtypes(include=np.number).columns.tolist()
    return render_template('prediction.html', 
                           target_feature=None, 
                           features=numerical_features, 
                           trained=False)

@app.route('/prediction_input', methods=['GET', 'POST'])
def prediction_input():
    global selected_model_type, model, selected_features, label_encoders, unique_values, X_train, y_train, X_test, y_test

    if not model:
        if selected_model_type == 'Linear Regression':
            model = LinearRegression()
        elif selected_model_type == 'Ridge Regression':
            model = Ridge()
        elif selected_model_type == 'SVR':
            model = SVR()
        elif selected_model_type == 'Random Forest':
            model = RandomForestRegressor()

    
    model.fit(X_train, y_train)

    scatter_plot_div = None 

    if request.method == 'POST':
        input_data = []

        # Collect input data
        for feature in selected_features:
            if feature in unique_values and unique_values[feature] is not None:
                input_value = request.form.get(feature)
                if input_value not in unique_values[feature]:
                    return render_template('prediction_input.html', 
                                           prediction=None, 
                                           scatter_plot=scatter_plot_div,
                                           error=f"Invalid input for {feature}")
                encoded_value = label_encoders[feature].transform([input_value])[0]
                input_data.append(encoded_value)
            else:
                try:
                    input_data.append(float(request.form[feature]))
                except ValueError:
                    return render_template('prediction_input.html', 
                                           prediction=None, 
                                           scatter_plot=scatter_plot_div,
                                           error=f"Invalid input for {feature}")

        input_df = pd.DataFrame([input_data], columns=selected_features)

        prediction_result = model.predict(input_df)[0].round(2)

        y_pred_test = model.predict(X_test)
        scatter_plot = go.Figure()
        scatter_plot.add_trace(
            go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='Predictions')
        )
        scatter_plot.add_trace(
            go.Scatter(
                x=y_test, 
                y=y_test, 
                mode='lines', 
                name='Actual', 
                line=dict(color='red', dash='dash')
            )
        )
        scatter_plot.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )
        scatter_plot_div = scatter_plot.to_html(full_html=False)

        return redirect(url_for('prediction_result', prediction=prediction_result))

    if X_test is not None and y_test is not None:
        y_pred_test = model.predict(X_test)
        scatter_plot = go.Figure()
        scatter_plot.add_trace(
            go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='Predictions')
        )
        scatter_plot.add_trace(
            go.Scatter(
                x=y_test, 
                y=y_test, 
                mode='lines', 
                name='Actual', 
                line=dict(color='red', dash='dash')
            )
        )
        scatter_plot.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )
        scatter_plot_div = scatter_plot.to_html(full_html=False)

    return render_template('prediction_input.html', 
                           selected_features=selected_features, 
                           unique_values=unique_values, 
                           scatter_plot=scatter_plot_div)

@app.route('/prediction_result', methods=['GET'])
def prediction_result():
    prediction_result = request.args.get('prediction', None)

    if prediction_result is None:
        return redirect(url_for('prediction_input'))  

    return render_template('prediction_result.html', prediction=prediction_result)


@app.route('/association_rules_view', methods=['GET', 'POST'])
def association_rules_view():
    global dataset_rules, unique_values

    skill_keywords = ["Python", "SQL", "Java", "Excel", "AWS", "Spark"]
    special_r_columns = ['r_yn', "r", "r_language", "r_lang"]
    
    skill_columns = []
    for col in dataset_rules.columns:
        if any(skill.lower() in col.lower() for skill in skill_keywords):
            skill_columns.append(col)
    for col in dataset_rules.columns:
        if col.lower() in special_r_columns:
            skill_columns.append(col)

    skill_data = dataset_rules[skill_columns].fillna(0)

    for col in skill_data.columns:
        skill_data[col] = skill_data[col].apply(lambda x: 1 if x else 0)

    frequent_itemsets = apriori(skill_data, min_support=0.01, use_colnames=True)


    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2, num_itemsets=10)

    rules_json = rules.to_json(orient="records")
    rules_data = json.loads(rules_json)


    unique_values = {
        'skills': skill_columns 
    }

    if request.method == 'POST':
        selected_skills = request.form.getlist('skills') 
        
        if not selected_skills:
            return render_template('association_rules.html', rules=[], unique_values=unique_values, error="Please select at least one skill.")
        
        selected_skills_set = set(selected_skills)  
        
        filtered_rules = []
        for rule in rules_data:
            antecedents = set(rule['antecedents']) 
            consequents = rule['consequents']
            
            if antecedents == selected_skills_set:
                filtered_rules.append(rule)

        return render_template('association_rules.html', rules=filtered_rules, unique_values=unique_values)

    return render_template('association_rules.html', rules=rules_data, unique_values=unique_values)

if __name__ == '__main__':
    app.run(debug=True)
