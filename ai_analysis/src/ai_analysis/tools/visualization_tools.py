from crewai.tools import BaseTool
from typing import Type, Union
from pydantic import BaseModel, Field
from src.ai_analysis.tools.data_manager import DataManager
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

PLOTS_DIR = Path("knowledge/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class VisualizationBaseTool(BaseTool):
    """Base class for visualization tools using cleaned data"""
    def _get_clean_data(self) -> Union[pd.DataFrame, str]:
        try:
            df = DataManager.get_df(processed=True)
            
            if df is None:
                return "No data available. Load and process data first."
            if df.empty:
                return "Cleaned data is empty. Check data cleaning steps."
                
            return df
            
        except Exception as e:
            return f"Data loading failed: {str(e)}"

# --- 1. Fixed Bar Chart Implementation ---
class BarChartInput(BaseModel):
    filename: str = Field(
        default="avg_price_by_category.png",
        description="Filename for saving the chart image"
    )

class BarChartAvgPriceTool(VisualizationBaseTool):
    name: str = "Price by Category Chart"
    description: str = "Creates bar chart of average prices using cleaned data"
    args_schema: Type[BaseModel] = BarChartInput

    def _run(self, filename: str) -> str:
        data = self._get_clean_data()
        if isinstance(data, str):
            return data

        try:
            if 'Category' not in data.columns or 'Price' not in data.columns:
                return "Missing required columns (Category/Price)"

            # Ensure correct filename
            filename = Path(filename).name  # Full filename with extension
            path = PLOTS_DIR / filename

            plt.figure(figsize=(12, 7))
            ax = sns.barplot(
                data=data,
                x="Category",
                y="Price",
                hue="Category",
                estimator='mean',
                errorbar='sd',
                palette='viridis',
                legend=False
            )
            plt.title("Price Distribution by Category (Cleaned Data)\nwith Standard Deviation", pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("Product Category", labelpad=15)
            plt.ylabel("Average Price", labelpad=15)

            if ax.containers:
                ax.bar_label(ax.containers[0], fmt='€%.2f')

            plt.tight_layout()

            # Ensure directory exists before saving
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)

            plt.savefig(path, dpi=300, bbox_inches='tight')
            return f"✅ Success: Chart saved to {path}"

        except Exception as e:
            return f"Chart creation failed: {str(e)}"
        finally:
            plt.close('all')


# --- 2. Enhanced Scatter Plot with Fixes ---
class ScatterPlotInput(BaseModel):
    filename: str = Field(
        default="price_rating_relation.png",
        description="Filename for saving the scatter plot"
    )

class ScatterPlotTool(VisualizationBaseTool):
    name: str = "Price-Rating Relationship"
    description: str = "Visualizes price vs rating correlation with regression line"
    args_schema: Type[BaseModel] = ScatterPlotInput

    def _run(self, filename: str) -> str:
        data = self._get_clean_data()
        if isinstance(data, str):
            return data

        try:
            filename = Path(filename).stem + ".png"
            path = PLOTS_DIR / filename

            # Initialize the figure
            plt.figure(figsize=(10, 6))

            # Plot each category separately with regression line
            categories = data['Category'].dropna().unique()
            for category in categories:
                subset = data[data['Category'] == category]
                sns.regplot(
                    data=subset,
                    x="Price",
                    y="Rating",
                    label=category,
                    scatter_kws={'alpha': 0.6, 's': 40},
                    line_kws={'linewidth': 1}
                )

            plt.title("Price vs Rating Correlation with Regression Lines", pad=15)
            plt.xlabel("Price (€)", labelpad=10)
            plt.ylabel("Customer Rating", labelpad=10)
            plt.legend(title="Category")
            plt.tight_layout()
            plt.savefig(path, dpi=300)

            return f"✅ Success: Scatter plot saved to {path}"

        except Exception as e:
            return f"Scatter plot failed: {str(e)}"
        finally:
            plt.close()

# --- 3. Fixed Top Rated Brands Chart ---
class BarChartTopRatedBrandsInput(BaseModel):
    top_n: int = Field(default=5, description="Top N brands to show")
    filename: str = Field(default="top_rated_brands.png")

class BarChartTopRatedBrandsTool(VisualizationBaseTool):
    name: str = "Top Rated Brands Chart"
    description: str = "Bar chart of top N brands by average customer rating"
    args_schema: Type[BaseModel] = BarChartTopRatedBrandsInput

    def _run(self, top_n: int, filename: str) -> str:
        data = self._get_clean_data()
        if isinstance(data, str):
            return data

        try:
            if 'Brand' not in data.columns or 'Rating' not in data.columns:
                return "Missing required columns (Brand/Rating)"

            result = (
                data.groupby("Brand")["Rating"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )

            if result.empty:
                return "No brands found with rating data"

            filename = Path(filename).stem + ".png"
            path = PLOTS_DIR / filename

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x="Brand",
                y="Rating",
                hue="Brand",
                data=result,
                palette='crest',
                legend=False
            )
            ax.set_title(f"Top {top_n} Brands by Rating", pad=15)
            ax.set_ylabel("Average Rating")
            ax.set_xlabel("Brand")

            if ax.containers:
                ax.bar_label(ax.containers[0], fmt="%.2f")

            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')

            return f"✅ Success: Chart saved to {path}"
        except Exception as e:
            return f"Failed to generate chart: {str(e)}"
        finally:
            plt.close('all')

# --- 4. Fixed Product Count Chart ---
class BarChartProductCountInput(BaseModel):
    filename: str = Field(default="product_count_by_brand.png")

class BarChartProductCountByBrandTool(VisualizationBaseTool):
    name: str = "Product Count by Brand"
    description: str = "Visualizes product distribution across brands"
    args_schema: Type[BaseModel] = BarChartProductCountInput

    def _run(self, filename: str) -> str:
        data = self._get_clean_data()
        if isinstance(data, str):
            return data

        try:
            if 'Brand' not in data.columns:
                return "Missing required column: Brand"

            counts = data["Brand"].value_counts()
            if counts.empty:
                return "No brand data available for visualization"

            filename = Path(filename).stem + ".png"
            path = PLOTS_DIR / filename

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x=counts.index,
                y=counts.values,
                hue=counts.index,
                palette='pastel',
                legend=False
            )
            ax.set_title("Product Count by Brand", pad=15)
            ax.set_ylabel("Product Count")
            ax.set_xlabel("Brand")

            if ax.containers:
                ax.bar_label(ax.containers[0])

            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')

            return f"✅ Success: Chart saved to {path}"
        except Exception as e:
            return f"Failed to generate chart: {str(e)}"
        finally:
            plt.close('all')