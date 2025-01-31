from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Import once at the top
import seaborn as sns
import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import base64
from io import BytesIO
import traceback

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key"

# Set up folders with correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Professional color scheme
COLOR_SCHEME = {
    'primary': '#1976d2',     # Primary blue
    'secondary': '#64b5f6',   # Light blue
    'accent': '#0d47a1',      # Dark blue
    'background': '#f5f5f5',  # Light gray background
    'text': '#212121',        # Dark text
    'border': '#e0e0e0',      # Border gray
    'success': '#43a047',     # Success green
    'warning': '#fb8c00',     # Warning orange
    'error': '#e53935',       # Error red
    'header': '#f8f9fa',      # Header background
    'table_hover': '#f5f5f5'  # Table hover color
}

# Gradient colors for efficiency visualization
EFFICIENCY_COLORS = [
    '#ffffff',  # White
    '#f3f6f9',  # Lightest blue
    '#e3f2fd',
    '#bbdefb',
    '#90caf9',
    '#64b5f6',
    '#42a5f5',
    '#2196f3',
    '#1e88e5',
    '#1976d2'   # Darkest blue
]

# Define operator name standardization mapping
OPERATOR_MAPPING = {
    'Mustafa Mulla': 'Mustafa Mulla',
    'Mustafa mulla': 'Mustafa Mulla',
    'Mustafa M': 'Mustafa Mulla',
    'Abhishek kodak': 'Abhishek Kodak',
    'Abhishek Koadak': 'Abhishek Kodak',
    'Abhishek K': 'Abhishek Kodak',
    'Eknath Patil': 'Eknath Patil',
    'Eknath P': 'Eknath Patil',
    'Ram Dhabale': 'Ram Dhabale',
    'R Dhabale': 'Ram Dhabale',
    'Rohidas Patil': 'Rohidas Patil',
    'R Patil': 'Rohidas Patil',
    'Suraj Patil': 'Suraj Patil',
    'S Patil': 'Suraj Patil',
    'Harsh Huddale': 'Harsh Huddale',
    'H Huddale': 'Harsh Huddale',
    'Prakash Patil': 'Prakash Patil',
    'P Patil': 'Prakash Patil'
}

custom_colormap = LinearSegmentedColormap.from_list('custom_blues', EFFICIENCY_COLORS)

def clean_dataframe(df):
    """Clean and prepare the dataframe for analysis"""
    try:
        df = df.copy()
        
        # Remove completely empty rows
        df = df.dropna(subset=['Operator Name', 'Part Name ', 'Part no.', 'OK Qty.', 'Prod. Qty.'], how='all')
        
        # Clean strings and standardize formatting
        string_columns = ['Operator Name', 'Part Name ', 'Part no.']
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
            # Remove any leading/trailing whitespace and convert to title case
            df[col] = df[col].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
        
        # Standardize operator names using global mapping
        df['Operator Name'] = df['Operator Name'].replace(OPERATOR_MAPPING)
        
        # Convert numeric columns
        numeric_columns = ['OK Qty.', 'Prod. Qty.']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Remove negative values
            df[col] = df[col].apply(lambda x: max(0, x))
        
        # Clean percentage columns
        def clean_percentage(x):
            if pd.isna(x):
                return 95.0  # Default value for missing percentages
            try:
                # Remove any % sign and convert to float
                value = float(str(x).replace('%', ''))
                # Ensure percentage is between 0 and 100
                return max(0, min(100, value))
            except:
                return 95.0
        
        # Process percentage columns
        percentage_columns = ['Performacne Rate', 'Availability']
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_percentage)
            else:
                df[col] = 95.0  # Default value if column is missing
        
        # Remove invalid data combinations
        df = df[
            (df['OK Qty.'] <= df['Prod. Qty.']) &  # OK quantity can't exceed total production
            (df['Prod. Qty.'] > 0) &  # Must have some production
            (df['OK Qty.'] >= 0) &  # Can't have negative quantities
            (df['Performacne Rate'] <= 100) &  # Can't exceed 100%
            (df['Availability'] <= 100)  # Can't exceed 100%
        ]
        
        # Add calculated columns
        df['Quality Rate'] = (df['OK Qty.'] / df['Prod. Qty.'] * 100).round(2)
        df['Efficiency'] = ((df['Quality Rate'] * 0.6 + 
                           df['Performacne Rate'] * 0.2 + 
                           df['Availability'] * 0.2)).round(2)
        
        return df.sort_values(['Part Name ', 'Part no.', 'Operator Name'])
        
    except Exception as e:
        print(f"Error cleaning dataframe: {str(e)}")
        traceback.print_exc()
        return df

def calculate_operator_efficiency(df, operator, part_name=None, part_no=None):
    """Calculate operator efficiency with improved accuracy and validation"""
    try:
        operator = str(operator).strip()
        operator_data = df[df['Operator Name'] == operator].copy()
        
        if part_name:
            operator_data = operator_data[operator_data['Part Name '].str.strip() == str(part_name).strip()]
        if part_no:
            operator_data = operator_data[operator_data['Part no.'].str.strip() == str(part_no).strip()]
        
        if len(operator_data) == 0:
            return 0.0
        
        # Basic metrics calculation
        total_ok = operator_data['OK Qty.'].sum()
        total_prod = operator_data['Prod. Qty.'].sum()
        quality_rate = (total_ok / total_prod * 100) if total_prod > 0 else 0
        performance_rate = operator_data['Performacne Rate'].mean()
        availability_rate = operator_data['Availability'].mean()
        
        # Component weights for efficiency calculation
        weights = {
            'quality': 0.6,       # Quality is most important
            'performance': 0.2,   # Performance and availability
            'availability': 0.2   # share remaining weight
        }
        
        # Base efficiency calculation
        base_efficiency = (
            quality_rate * weights['quality'] +
            performance_rate * weights['performance'] +
            availability_rate * weights['availability']
        )
        
        # Experience bonus calculation
        total_parts_produced = operator_data['Prod. Qty.'].sum()
        experience_level = min(5, np.log1p(total_parts_produced) / 10)  # Logarithmic scaling
        
        # Consistency bonus calculation
        if len(operator_data) > 1:
            # Calculate coefficient of variation for quality
            quality_mean = operator_data['Quality Rate'].mean()
            quality_std = operator_data['Quality Rate'].std()
            cv = quality_std / quality_mean if quality_mean > 0 else 1
            
            # More consistent = lower CV = higher bonus
            consistency_bonus = 5 * (1 - min(1, cv))
        else:
            consistency_bonus = 0
        
        # Performance trend bonus
        if len(operator_data) > 2:
            # Sort by date and calculate trend
            operator_data['Quality Rate'] = operator_data['OK Qty.'] / operator_data['Prod. Qty.'] * 100
            recent_trend = operator_data['Quality Rate'].tail(3).mean() - operator_data['Quality Rate'].head(3).mean()
            trend_bonus = min(2, max(0, recent_trend / 10))  # Up to 2% bonus for positive trend
        else:
            trend_bonus = 0
        
        # Calculate final efficiency with bonuses
        total_bonus = experience_level + consistency_bonus + trend_bonus
        final_efficiency = base_efficiency * (1 + total_bonus / 100)
        
        # Ensure result is within reasonable bounds
        bounded_efficiency = min(100, max(60, final_efficiency))
        
        return bounded_efficiency
        
    except Exception as e:
        print(f"Error calculating efficiency for {operator}: {str(e)}")
        traceback.print_exc()
        return 0.0

def create_matrices_from_data(df):
    """Create operator matrices with improved calculations"""
    try:
        # Get unique operators and parts, sorted for consistency
        operators = sorted(df['Operator Name'].unique())
        parts = sorted(df['Part Name '].unique())
        
        if not operators or not parts:
            print("No valid operators or parts found")
            return pd.DataFrame(), pd.DataFrame()
        
        # Initialize matrices with zeros
        operator_vs_part = pd.DataFrame(0.0, index=operators, columns=parts)
        operator_vs_operation = pd.DataFrame(0.0, index=operators, columns=parts)
        
        # Calculate and fill matrices
        for operator in operators:
            operator_data = df[df['Operator Name'] == operator]
            for part in parts:
                # Get part-specific data
                part_data = operator_data[operator_data['Part Name '] == part]
                
                if not part_data.empty:
                    # Calculate part efficiency using the main efficiency function
                    part_efficiency = calculate_operator_efficiency(df, operator, part_name=part)
                    
                    # Calculate operation efficiency using component metrics
                    operation_metrics = {
                        'performance': part_data['Performacne Rate'].mean(),
                        'availability': part_data['Availability'].mean(),
                        'quality': (part_data['OK Qty.'].sum() / part_data['Prod. Qty.'].sum() * 100) 
                            if part_data['Prod. Qty.'].sum() > 0 else 0
                    }
                    
                    # Weight the operation metrics differently
                    operation_efficiency = (
                        operation_metrics['performance'] * 0.4 +  # Performance is key for operations
                        operation_metrics['availability'] * 0.3 + # Availability is important
                        operation_metrics['quality'] * 0.3       # Quality maintains importance
                    )
                    
                    # Store calculated efficiencies in respective matrices
                    operator_vs_part.loc[operator, part] = part_efficiency
                    operator_vs_operation.loc[operator, part] = operation_efficiency
        
        return operator_vs_part, operator_vs_operation
        
    except Exception as e:
        print(f"Error creating matrices: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def get_part_mapping_from_data(df):
    """Create dynamic part mapping from cleaned data"""
    part_mapping = {}
    try:
        # Ensure data is properly formatted
        df['Part Name '] = df['Part Name '].astype(str).str.strip()
        df['Part no.'] = df['Part no.'].astype(str).str.strip()
        
        # Get unique parts and their numbers
        for part_name in df['Part Name '].unique():
            if pd.isna(part_name) or part_name == 'nan':
                continue
                
            part_data = df[df['Part Name '].str.strip() == part_name.strip()]
            part_numbers = [str(num).strip() for num in part_data['Part no.'].unique() 
                          if pd.notna(num) and str(num).strip() != 'nan']
            
            if part_numbers:
                part_mapping[part_name.strip()] = sorted(set(part_numbers))  # Use set to remove duplicates
                
    except Exception as e:
        print(f"Error creating part mapping: {str(e)}")
        traceback.print_exc()
        part_mapping = {}
    
    return part_mapping

def generate_dynamic_allocations(df, absent_operators):
    """Generate optimized allocations with improved efficiency calculation"""
    try:
        # Clean and prepare data
        df['Operator Name'] = df['Operator Name'].astype(str).str.strip()
        df['Part Name '] = df['Part Name '].astype(str).str.strip()
        df['Part no.'] = df['Part no.'].astype(str).str.strip()
        
        # Initialize results storage
        allocation_results = []
        available_operators = {str(op).strip() for op in df['Operator Name'].unique() 
                             if pd.notna(op)} - {str(op).strip() for op in absent_operators}
        allocated_operators = set()
        
        # Sort parts by complexity (using number of unique operators as proxy)
        part_complexity = {}
        for part_name in sorted(df['Part Name '].unique()):
            if pd.isna(part_name) or part_name == 'nan':
                continue
            part_data = df[df['Part Name '].str.strip() == part_name.strip()]
            part_complexity[part_name] = len(part_data['Operator Name'].unique())
        
        # Process parts in order of complexity (most complex first)
        for part_name in sorted(part_complexity.keys(), key=lambda x: -part_complexity[x]):
            part_data = df[df['Part Name '].str.strip() == part_name.strip()]
            
            for subpart in sorted(part_data['Part no.'].unique()):
                if pd.isna(subpart) or str(subpart).strip() == 'nan':
                    continue
                
                # Get efficiency scores for available operators
                operator_scores = {}
                for operator in available_operators - allocated_operators:
                    efficiency = calculate_operator_efficiency(
                        df, operator, 
                        part_name=part_name, 
                        part_no=subpart
                    )
                    if efficiency > 0:
                        operator_scores[operator] = efficiency
                
                if operator_scores:
                    # Find best operator considering both efficiency and workload
                    best_operator = max(operator_scores.items(), key=lambda x: x[1])
                    allocation_results.append({
                        "Part": part_name.strip(),
                        "Sub-Part": str(subpart).strip(),
                        "Operator": best_operator[0],
                        "Efficiency": round(best_operator[1], 2)  # Round to 2 decimal places
                    })
                    allocated_operators.add(best_operator[0])
                else:
                    allocation_results.append({
                        "Part": part_name.strip(),
                        "Sub-Part": str(subpart).strip(),
                        "Operator": "NO AVAILABLE OPERATOR",
                        "Efficiency": 0.0
                    })
        
        if not allocation_results:
            return "<p>No allocation data available</p>"
        
        # Create and style DataFrame
        df_allocations = pd.DataFrame(allocation_results)        
        # Style the allocations table
        styled_df = df_allocations.style\
            .format({
                'Part': str,
                'Sub-Part': str,
                'Operator': str,
                'Efficiency': '{:.2f}%'
            })\
            .background_gradient(
                subset=['Efficiency'],
                cmap=custom_colormap,
                vmin=60,
                vmax=100
            )\
            .set_properties(**{
                'text-align': 'center',
                'padding': '8px',
                'border': f'1px solid {COLOR_SCHEME["border"]}'
            })\
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', COLOR_SCHEME['header']),
                    ('color', COLOR_SCHEME['text']),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                    ('border', f'1px solid {COLOR_SCHEME["border"]}')
                ]},
                {'selector': 'td', 'props': [
                    ('text-align', 'center'),
                    ('padding', '8px'),
                    ('border', f'1px solid {COLOR_SCHEME["border"]}')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', COLOR_SCHEME['table_hover'])
                ]},
                {'selector': 'table', 'props': [
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('margin-bottom', '1rem'),
                    ('background-color', 'white'),
                    ('border-radius', '8px'),
                    ('overflow', 'hidden'),
                    ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
                ]}
            ])
        
        return styled_df.to_html(classes='table table-hover', escape=False)
        
    except Exception as e:
        print(f"Error in generate_dynamic_allocations: {str(e)}")
        traceback.print_exc()
        return "<p>Error generating allocations</p>"
    
@app.route('/')
def index():
    """Render the upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return redirect(url_for('index'))
        
    """Handle file upload and initial data processing"""
    try:
        # Validate file upload
        if 'file' not in request.files:
            flash('Please select a file to upload')
            return redirect(url_for('index'))
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))
            
        if not file.filename.endswith('.csv'):
            flash('Please upload a CSV file')
            return redirect(url_for('index'))
        
        if file:
            # Clear previous uploads
            for f in os.listdir(UPLOAD_FOLDER):
                if f.endswith('.csv'):
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
            
            # Save and process new file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Read and clean the data
            df = pd.read_csv(file_path)
            cleaned_df = clean_dataframe(df)
            
            if cleaned_df.empty:
                flash('Error: No valid data found in the uploaded file')
                return redirect(url_for('index'))
            
            # Save cleaned data
            cleaned_path = os.path.join(UPLOAD_FOLDER, 'cleaned_' + file.filename)
            cleaned_df.to_csv(cleaned_path, index=False)
            
            # Create matrices from cleaned data
            operator_vs_part, operator_vs_operation = create_matrices_from_data(cleaned_df)
            
            if operator_vs_part.empty or operator_vs_operation.empty:
                flash('Error: Unable to create performance matrices')
                return redirect(url_for('index'))
                        
            # Style matrices
            matrix_style = [
                {'selector': 'th', 'props': [
                    ('background-color', COLOR_SCHEME['header']),
                    ('color', COLOR_SCHEME['text']),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                    ('border', f'1px solid {COLOR_SCHEME["border"]}')
                ]},
                {'selector': 'td', 'props': [
                    ('text-align', 'center'),
                    ('padding', '8px'),
                    ('border', f'1px solid {COLOR_SCHEME["border"]}')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', COLOR_SCHEME['table_hover'])
                ]},
                {'selector': 'table', 'props': [
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('margin-bottom', '1rem'),
                    ('background-color', 'white'),
                    ('border-radius', '8px'),
                    ('overflow', 'hidden'),
                    ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
                ]}
            ]
            
            # Style the operator vs part matrix
            styled_op_part = operator_vs_part.style\
                .background_gradient(
                    cmap=custom_colormap,
                    vmin=60,
                    vmax=100
                )\
                .format("{:.1f}%")\
                .set_table_styles(matrix_style)

            # Style the operator vs operation matrix
            styled_op_operation = operator_vs_operation.style\
                .background_gradient(
                    cmap=custom_colormap,
                    vmin=60,
                    vmax=100
                )\
                .format("{:.1f}%")\
                .set_table_styles(matrix_style)
            
            # Get part mapping and generate allocations
            part_mapping = get_part_mapping_from_data(cleaned_df)
            initial_allocations = generate_dynamic_allocations(cleaned_df, set())
            
            if initial_allocations == "<p>Error generating allocations</p>":
                flash('Error generating allocations')
                return redirect(url_for('index'))
            
            # Log statistics for verification
            print("\nData Statistics after processing:")
            print(f"Total rows: {len(cleaned_df)}")
            print(f"Unique operators: {len(cleaned_df['Operator Name'].unique())}")
            print(f"Unique parts: {len(cleaned_df['Part Name '].unique())}")
            print(f"Average efficiency: {operator_vs_part.mean().mean():.2f}%")
            
            return render_template(
                'results.html',
                operator_vs_part=styled_op_part.to_html(classes='table table-hover', escape=False),
                operator_vs_operation=styled_op_operation.to_html(classes='table table-hover', escape=False),
                initial_dynamic_allocations=initial_allocations,
                parts_list=sorted(part_mapping.keys()),
                parts_with_numbers=part_mapping,
                color_scheme=COLOR_SCHEME
            )
            
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        traceback.print_exc()
        flash('Error occurred during file processing')
        return redirect(url_for('index'))
   
@app.route('/get_subparts', methods=['POST'])
def get_subparts():
    """Get sub-parts for a selected part"""
    try:
        selected_part = request.form.get('part')
        if not selected_part:
            return jsonify({
                'status': 'error',
                'message': 'No part selected'
            })
        
        # Get the cleaned CSV file
        cleaned_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('cleaned_') and f.endswith('.csv')]
        if not cleaned_files:
            return jsonify({
                'status': 'error',
                'message': 'No cleaned CSV file found'
            })
            
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, cleaned_files[0]))
        part_mapping = get_part_mapping_from_data(df)
        
        subparts = part_mapping.get(selected_part, [])
        return jsonify({
            'status': 'success',
            'subparts': subparts
        })
    except Exception as e:
        print(f"Error in get_subparts: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/get_part_performance', methods=['POST'])
def get_part_performance():
    """Get performance data for a specific part and sub-part"""
    try:
        selected_part = request.form.get('part')
        selected_subpart = request.form.get('subpart')
        
        if not selected_part:
            return jsonify({
                'status': 'error',
                'message': 'No part selected'
            })

        # Get the cleaned CSV file
        cleaned_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('cleaned_') and f.endswith('.csv')]
        if not cleaned_files:
            return jsonify({
                'status': 'error',
                'message': 'No cleaned CSV file found'
            })
            
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, cleaned_files[0]))
        
        # Calculate efficiency for each operator
        operator_scores = {}
        for operator in df['Operator Name'].unique():
            efficiency = calculate_operator_efficiency(
                df, operator, 
                part_name=selected_part, 
                part_no=selected_subpart
            )
            if efficiency > 0:
                operator_scores[operator] = efficiency

        if not operator_scores:
            return jsonify({
                'status': 'error',
                'message': 'No data available for selected combination'
            })

        scores = pd.Series(operator_scores).sort_values(ascending=False)

        # Create visualization with professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # Create custom colormap
        norm = plt.Normalize(60, 100)
        colors = custom_colormap(norm(scores.values))
        
        bars = ax.bar(range(len(scores)), scores.values, color=colors)
        
        # Customize plot appearance
        ax.set_title(f'Operator Performance for {selected_part}\nSub-part: {selected_subpart}', 
                    pad=20, fontsize=14, color=COLOR_SCHEME['text'])
        ax.set_xlabel('Operators', fontsize=12, color=COLOR_SCHEME['text'])
        ax.set_ylabel('Efficiency Score (%)', fontsize=12, color=COLOR_SCHEME['text'])
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(scores.index, rotation=45, ha='right')
        
        # Set background and grid style
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%',
                   ha='center', va='bottom',
                   color=COLOR_SCHEME['text'],
                   fontsize=10)

        plt.tight_layout()
        
        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close('all')
        
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Get detailed stats for top performers
        top_operators = []
        for op, score in scores.head(3).items():
            op_data = df[
                (df['Operator Name'] == op) & 
                (df['Part Name '] == selected_part) &
                (df['Part no.'] == selected_subpart)
            ]
            
            # Calculate detailed metrics
            quality_rate = (op_data['OK Qty.'].sum() / op_data['Prod. Qty.'].sum() * 100) \
                if op_data['Prod. Qty.'].sum() > 0 else 0
            
            performance_rate = op_data['Performacne Rate'].mean()
            availability = op_data['Availability'].mean()
            
            top_operators.append({
                'operator': op,
                'score': f'{score:.1f}%',
                'quality_rate': f"{quality_rate:.1f}%",
                'performance_rate': f"{performance_rate:.1f}%",
                'availability': f"{availability:.1f}%",
                'experience': f"{len(op_data)} jobs completed",
                'total_production': f"{op_data['Prod. Qty.'].sum():,} units"
            })

        return jsonify({
            'status': 'success',
            'graph': f"data:image/png;base64,{graph_base64}",
            'top_operators': top_operators
        })

    except Exception as e:
        print(f"Error in get_part_performance: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/update_allocations', methods=['POST'])
def update_allocations():
    """Update allocations based on absent operators"""
    try:
        absent_operators = request.form.get('absent_operators', '')
        absent_operators = {op.strip() for op in absent_operators.split(',') if op.strip()}
        
        cleaned_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('cleaned_') and f.endswith('.csv')]
        if not cleaned_files:
            return jsonify({
                'status': 'error',
                'message': 'No cleaned CSV file found',
                'html': ''
            })
            
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, cleaned_files[0]))
        df['Operator Name'] = df['Operator Name'].astype(str).str.strip()
        
        invalid_operators = [op for op in absent_operators if op not in df['Operator Name'].unique()]
        if invalid_operators:
            return jsonify({
                'status': 'error',
                'message': f'Invalid operator names: {", ".join(invalid_operators)}',
                'html': ''
            })

        updated_allocations = generate_dynamic_allocations(df, absent_operators)
        
        return jsonify({
            'status': 'success',
            'message': 'Allocations updated successfully!',
            'html': updated_allocations
        })
    except Exception as e:
        print(f"Error in update_allocations: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}',
            'html': ''
        })

if __name__ == '__main__':
    app.run(debug=True)