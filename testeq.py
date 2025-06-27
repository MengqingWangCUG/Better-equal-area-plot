import numpy as np
import pandas as pd
import ultraplot as upt
from EQplot import equalareaplot, fisher_mean

def generate_test_data():
    """Generate test data"""
    # Basic test data - simulated paleomagnetic direction data
    np.random.seed(42)
    
    # Test dataset 1: scattered direction data
    dec1 = np.random.normal(350, 15, 10)  # declination, mean 350°, std 15°
    inc1 = np.random.normal(60, 10, 10)   # inclination, mean 60°, std 10°
    
    # Test dataset 2: more concentrated data
    dec2 = np.random.normal(45, 5, 8)
    inc2 = np.random.normal(-45, 5, 8)
    
    # Test dataset 3: single data point
    dec3 = [180]
    inc3 = [30]
    
    return {
        'scattered': {'dec': dec1, 'inc': inc1},
        'concentrated': {'dec': dec2, 'inc': inc2},
        'single': {'dec': dec3, 'inc': inc3}
    }

def test_basic_plot():
    """Test basic plotting functionality"""
    print("Test 1: Basic equal-area projection plot")
    
    data = generate_test_data()
    dec, inc = data['scattered']['dec'], data['scattered']['inc']
    
    # Create basic figure
    fig, ax = upt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax = equalareaplot(dec=dec, inc=inc, ax=ax, type=0)

    
    return True

def test_with_confidence_ellipse():
    """Test plotting with confidence ellipse"""
    print("Test 2: Equal-area projection plot with confidence ellipse")
    
    data = generate_test_data()
    dec, inc = data['concentrated']['dec'], data['concentrated']['inc']
    
    # Calculate a95 for each point (using simulated values here)
    a95 = np.random.uniform(5, 15, len(dec))
    
    fig, ax = upt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax = equalareaplot(dec=dec, inc=inc, a95=a95, ax=ax, type=0, 
                      cirlabel="Test Data", circolor='red', markersize=6)
    

    return True

def test_fisher_statistics():
    """Test Fisher statistics functionality"""
    print("Test 3: Fisher statistics analysis")
    
    data = generate_test_data()
    dec, inc = data['concentrated']['dec'], data['concentrated']['inc']
    
    fig, ax = upt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax = equalareaplot(dec=dec, inc=inc, ax=ax, type=0,
                      fisher=True, fishertextloc='r',
                      starlabel="Fisher Mean", starcolor='blue',
                      markersize=5)
    
    return True

def test_different_types():
    """Test different types of grids"""
    print("Test 4: Different grid types comparison")
    
    try:
        data = generate_test_data()
        dec, inc = data['scattered']['dec'], data['scattered']['inc']
        
        fig, axes = upt.subplots(ncols=2, figsize=(16, 8), 
                                subplot_kw=dict(projection='polar'))
        
        # Type 0 grid
        axes[0] = equalareaplot(dec=dec, inc=inc, ax=axes[0], type=0, 
                               legendbool=True, markersize=5)
        axes[0].set_title('Type 0 Grid')
        
        # Type 1 grid
        axes[1] = equalareaplot(dec=dec, inc=inc, ax=axes[1], type=1, 
                               showticks=True, legendbool=True, markersize=5)
        axes[1].set_title('Type 1 Grid')
        
        return True
    except Exception as e:
        print(f"Test 4 error details: {str(e)}")
        return False

def test_mixed_inclinations():
    """Test mixed inclination data (positive and negative inclinations)"""
    print("Test 5: Mixed inclination data")
    
    # Create data containing positive and negative inclinations
    dec_mixed = [0, 45, 90, 135, 180, 225, 270, 315]
    inc_mixed = [30, -45, 60, -30, 15, -60, 45, -15]
    
    fig, ax = upt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax = equalareaplot(dec=dec_mixed, inc=inc_mixed, ax=ax, type=0,
                      line=True, linewidth=2, markersize=8,
                      cirlabel="Upper and Lower Hemisphere Data", circolor='green')

    
    return True

def test_single_point():
    """Test single point data"""
    print("Test 6: Single point data")
    
    data = generate_test_data()
    dec, inc = data['single']['dec'], data['single']['inc']
    
    fig, ax = upt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax = equalareaplot(dec=dec, inc=inc, ax=ax, type=0,
                      a95=[20], markersize=10, circolor='purple')
    
    return True

def test_custom_fisher_data():
    """Test custom Fisher data"""
    print("Test 7: Custom Fisher statistics data")
    
    data = generate_test_data()
    dec, inc = data['concentrated']['dec'], data['concentrated']['inc']
    
    # Create custom Fisher results
    fisher_result = fisher_mean(dec=dec, inc=inc)
    
    fig, ax = upt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax = equalareaplot(dec=dec, inc=inc, ax=ax, type=0,
                      fisher=True, fisherdf=fisher_result,
                      fishertextloc='l', starlabel="Custom Fisher",
                      starcolor='orange', markersize=4)
    
    return True

def run_all_tests():
    """Run all tests"""
    print("Starting equal-area projection plot tests...")
    print("=" * 50)
    
    tests = [
        test_basic_plot,
        test_with_confidence_ellipse,
        test_fisher_statistics,
        test_different_types,
        test_mixed_inclinations,
        test_single_point,
        test_custom_fisher_data
    ]
    
    results = []
    for i, test_func in enumerate(tests, 1):
        try:
            result = test_func()
            if result:
                results.append(f"Test {i}: Passed")
                print(f"Test {i}: Passed")
            else:
                results.append(f"Test {i}: Failed")
                print(f"Test {i}: Failed")
        except Exception as e:
            results.append(f"Test {i}: Failed - {str(e)}")
            print(f"Test {i}: Failed - {str(e)}")
        print("-" * 30)
    
    print("\nTest Summary:")
    for result in results:
        print(result)
    
    return results

if __name__ == "__main__":
    # Set font support
    try:
        upt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        upt.rcParams['axes.unicode_minus'] = False
    except:
        print("Font setting failed, using default font")
    
    run_all_tests()