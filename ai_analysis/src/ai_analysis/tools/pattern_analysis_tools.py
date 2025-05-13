from crewai.tools import BaseTool
from typing import Type, Union
from pydantic import BaseModel, Field
from src.ai_analysis.tools.data_manager import DataManager
from pathlib import Path


OUTPUT_DIR = Path("knowledge/patterns")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save result function
def save_result(name: str, content: str) -> str:
    path = OUTPUT_DIR / f"{name}.txt"
    try:
        path.write_text(content)
        if path.exists():
            return f"✅ Result saved to {path}"
        else:
            return f"❌ Failed to save result to {path}"
    except Exception as e:
        return f"❌ Error while saving result: {str(e)}"

# --- 1. Top Rated Brands ---
class TopRatedBrandsTool(BaseTool):
    name: str = "Top Rated Brands Tool"
    description: str = "Returns the top N brands by average customer rating"
    
    class Args(BaseModel):
        top_n: int = Field(..., description="Number of top brands to return")
        
    def _run(self, top_n: int) -> str:
        try:
            df = DataManager.get_df(processed=True)
            
            # Check if the necessary columns are present
            if 'Brand' not in df.columns or 'Rating' not in df.columns:
                return "❌ Missing required columns (Brand/Rating)"
            
            result = (
                df.groupby("Brand")["Rating"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .round(2)
            )
            output = f"Top {top_n} brands by rating:\n{result.to_string()}"
            return save_result("top_rated_brands", output)
        
        except Exception as e:
            return f"❌ Brand analysis failed: {str(e)}. Verify data cleaning was completed."

# --- 2. Average Price by Category ---
class AvgPriceByCategoryTool(BaseTool):
    name: str = "Average Price By Category Tool"
    description: str = "Shows average product price in each fashion category"
    
    class Args(BaseModel):
        currency_symbol: str = Field("$", description="Currency symbol for display")
        
    def _run(self, currency_symbol: str = "$") -> str:
        try:
            df = DataManager.get_df(processed=True)
            
            # Check if the necessary columns are present
            if 'Category' not in df.columns or 'Price' not in df.columns:
                return "❌ Missing required columns (Category/Price)"
            
            result = df.groupby("Category")["Price"].mean().round(2)
            formatted = result.map(lambda x: f"{currency_symbol}{x}")
            output = f"Average prices by category:\n{formatted.to_string()}"
            return save_result("avg_price_by_category", output)
        
        except Exception as e:
            return f"❌ Price analysis failed: {str(e)}. Check numeric columns exist."

# --- 3. Most Common Color per Category ---
class TopColorPerCategoryTool(BaseTool):
    name: str = "Top Color Per Category Tool"
    description: str = "Finds the most frequent product color in each category"
    
    class Args(BaseModel):
        min_count: int = Field(5, description="Minimum occurrences to consider")
        
    def _run(self, min_count: int = 5) -> str:
        try:
            df = DataManager.get_df(processed=True)
            
            # Check if the necessary columns are present
            if 'Category' not in df.columns or 'Color' not in df.columns:
                return "❌ Missing required columns (Category/Color)"
            
            def get_top_color(x):
                counts = x.value_counts()
                return counts[counts >= min_count].idxmax() if not counts.empty else "N/A"
                
            result = df.groupby("Category")["Color"].agg(get_top_color)
            output = f"Most common colors by category (min {min_count} entries):\n{result.to_string()}"
            return save_result("top_color_per_category", output)
        
        except Exception as e:
            return f"❌ Color analysis failed: {str(e)}. Verify color data exists."

# --- 4. Product Count by Brand ---
class ProductCountByBrandTool(BaseTool):
    name: str = "Product Count By Brand Tool"
    description: str = "Returns the number of products available for each brand"
    
    class Args(BaseModel):
        sort_desc: bool = Field(True, description="Sort results descending")
        
    def _run(self, sort_desc: bool = True) -> str:
        try:
            df = DataManager.get_df(processed=True)
            
            # Check if the necessary column is present
            if 'Brand' not in df.columns:
                return "❌ Missing required column (Brand)"
            
            # Count product occurrences per brand
            counts = df["Brand"].value_counts(sort=sort_desc)
            
            # If counts is empty, indicate no products for any brand
            if counts.empty:
                return "❌ No products found for any brand. Ensure your data is correctly cleaned."

            output = f"Product count by brand:\n{counts.to_string()}"
            
            # Save the result to file
            result_message = save_result("product_count_by_brand", output)
            return result_message
        
        except Exception as e:
            return f"❌ Brand count failed: {str(e)}. Ensure brand data is clean."



