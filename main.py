"""
機器學習視覺化主程式
執行所有視覺化腳本，生成完整的學習輔助圖片
"""

import os
import sys
import time


def print_header(text):
    """印出美化的標題"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_visualization(module_name, description):
    """執行視覺化模組"""
    print(f"\n>>> 正在執行: {description}")
    print(f">>> 模組: {module_name}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        # 動態導入模組
        if module_name.startswith('visualizations.'):
            module = __import__(module_name, fromlist=['main'])
        else:
            module = __import__(module_name)
        
        # 執行 main 函數
        module.main()
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ 完成! 耗時: {elapsed_time:.2f} 秒")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ 錯誤! 耗時: {elapsed_time:.2f} 秒")
        print(f"錯誤訊息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程式"""
    print_header("🎓 機器學習核心概念視覺化工具 🎓")
    
    print("此程式將生成所有機器學習核心概念的視覺化圖片")
    print("幫助你從零開始理解所有重要概念！")
    print("\n輸出位置:")
    print("  • 圖片: output/images/")
    print("  • 影片: output/videos/")
    
    # 確保輸出目錄存在
    os.makedirs('output/images', exist_ok=True)
    os.makedirs('output/videos', exist_ok=True)
    
    # 定義所有視覺化模組
    visualizations = [
        ('visualizations.quick_reference', '快速參考卡 (Quick Reference Card)'),
        ('visualizations.regression', '回歸分析 (LSE, Ridge, Lasso, GD, Newton)'),
        ('visualizations.distributions', '機率分佈 (Gaussian, Beta, Poisson, Gamma)'),
        ('visualizations.classification', '分類方法 (Naive Bayes, Logistic Regression)'),
        ('visualizations.em_algorithm', 'EM 演算法 (Gaussian Mixture Model)'),
    ]
    
    # 統計
    total = len(visualizations)
    success = 0
    failed = []
    
    start_time = time.time()
    
    # 執行所有視覺化
    for i, (module, description) in enumerate(visualizations, 1):
        print_header(f"步驟 {i}/{total}: {description}")
        
        if run_visualization(module, description):
            success += 1
        else:
            failed.append(description)
    
    # 總結
    total_time = time.time() - start_time
    
    print_header("📊 執行總結 📊")
    
    print(f"總共執行: {total} 個視覺化模組")
    print(f"成功: {success} 個 ✓")
    print(f"失敗: {len(failed)} 個 ✗")
    print(f"總耗時: {total_time:.2f} 秒")
    
    if failed:
        print("\n失敗的模組:")
        for item in failed:
            print(f"  • {item}")
    
    print("\n" + "="*70)
    
    if success == total:
        print("🎉 所有視覺化已成功生成！")
        print("請查看 output/images/ 目錄中的圖片")
        print("配合 ML_NOTES.md 學習效果最佳！")
    else:
        print("⚠️  部分視覺化生成失敗，請檢查錯誤訊息")
    
    print("="*70 + "\n")
    
    # 列出生成的檔案
    if os.path.exists('output/images'):
        images = sorted([f for f in os.listdir('output/images') if f.endswith('.png')])
        if images:
            print(f"\n已生成 {len(images)} 張圖片:")
            for img in images:
                print(f"  • {img}")
    
    return success == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
