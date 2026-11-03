import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule

def export_experiments_to_excel(cfg, df_a, df_b, df_c, optimal_info, filepath="logs/MLP_Experimental_Results.xlsx"):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active) # Xóa sheet mặc định

    # Bảng màu và Font 
    header_fill = PatternFill(start_color="2F4F4F", end_color="2F4F4F", fill_type="solid")
    zebra_fill = PatternFill(start_color="F4F7F9", end_color="F4F7F9", fill_type="solid")
    highlight_fill = PatternFill(start_color="E6F2EA", end_color="E6F2EA", fill_type="solid")

    font_title = Font(name="Calibri", size=13, bold=True, color="1B2631")
    font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=10, bold=True, color="1B2631")
    font_reg = Font(name="Calibri", size=10, color="1B2631")

    thin_border = Border(
        left=Side(style='thin', color='D3D3D3'),
        right=Side(style='thin', color='D3D3D3'),
        top=Side(style='thin', color='D3D3D3'),
        bottom=Side(style='thin', color='D3D3D3')
    )

    def write_dataframe_sheet(title, heading, df, num_cols, time_cols=None):
        if time_cols is None:
            time_cols = []
        ws = wb.create_sheet(title=title)
        ws.freeze_panes = "A4"
        ws["A1"] = heading
        ws["A1"].font = font_title

        headers = list(df.columns)
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col_idx, value=h)
            cell.fill = header_fill
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        ws.row_dimensions[3].height = 26

        start_row = 4
        for r_idx, row_vals in enumerate(df.values, start_row):
            ws.row_dimensions[r_idx].height = 20
            is_even = (r_idx - start_row) % 2 == 1
            for c_idx, val in enumerate(row_vals, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.font = font_reg
                cell.border = thin_border
                if is_even:
                    cell.fill = zebra_fill
                if c_idx in num_cols:
                    cell.number_format = "0.000000"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif c_idx in time_cols:
                    cell.number_format = "0.00"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="center", vertical="center")

        end_row = start_row + len(df) - 1
        # Thêm dòng Trung bình (AVERAGE) và Tốt nhất (MIN)
        avg_row = end_row + 1
        min_row = end_row + 2
        ws.cell(row=avg_row, column=1, value="Trung bình (Mean)").font = font_bold
        ws.cell(row=min_row, column=1, value="Tốt nhất (Min)").font = font_bold

        for c_idx in range(1, len(headers) + 1):
            col_ltr = get_column_letter(c_idx)
            if c_idx in num_cols or c_idx in time_cols:
                ws.cell(row=avg_row, column=c_idx, value=f"=AVERAGE({col_ltr}{start_row}:{col_ltr}{end_row})")
                ws.cell(row=min_row, column=c_idx, value=f"=MIN({col_ltr}{start_row}:{col_ltr}{end_row})")
                fmt = "0.000000" if c_idx in num_cols else "0.00"
                ws.cell(row=avg_row, column=c_idx).number_format = fmt
                ws.cell(row=min_row, column=c_idx).number_format = fmt
            ws.cell(row=avg_row, column=c_idx).font = font_bold
            ws.cell(row=min_row, column=c_idx).font = font_bold
            ws.cell(row=min_row, column=c_idx).fill = highlight_fill
            ws.cell(row=avg_row, column=c_idx).border = thin_border
            ws.cell(row=min_row, column=c_idx).border = thin_border
            ws.column_dimensions[col_ltr].width = max(len(str(headers[c_idx-1])) + 5, 14)

        # Tô màu nhiệt cột Validation MSE và Test MSE
        ws.conditional_formatting.add(
            f"F{start_row}:F{end_row}",
            ColorScaleRule(start_type="min", start_color="63BE7B", mid_type="percentile", mid_value=50, mid_color="FFEB84", end_type="max", end_color="F8696B")
        )
        return ws, start_row, end_row

    # 1. Sheet Tổng quan Baseline & Mô hình tối ưu
    ws_sum = wb.create_sheet(title="Tong_Quan_Baseline")
    ws_sum["A1"] = "TỔNG HỢP CẤU HÌNH BASELINE & MÔ HÌNH TỐI ƯU"
    ws_sum["A1"].font = font_title

    summary_rows = [
        ("Tham số / Chỉ số", "Cấu hình Baseline", "Mô hình Tối ưu (Optimal)"),
        ("Số mẫu huấn luyện (A)", cfg["train_size"], optimal_info["best_A"]),
        ("Kiến trúc tầng ẩn", str(cfg["hidden_layers"]), str(optimal_info["optimal_arch"])),
        ("Hàm kích hoạt (Activation)", cfg["activation"], cfg["activation"]),
        ("Bộ tối ưu & Learning Rate", f"{cfg['optimizer']} (lr={cfg['lr']})", f"{cfg['optimizer']} (lr={cfg['lr']})"),
        ("Batch Size / Epochs", f"{cfg['batch_size']} / {cfg['epochs']}", f"{cfg['batch_size']} / {cfg['epochs']}"),
        ("Random Seeds", str(cfg["seeds"]), str(cfg["seeds"])),
        ("Test MSE (N=100)", "-", round(optimal_info["test_metrics"]["mse"], 6)),
        ("Test MAE (N=100)", "-", round(optimal_info["test_metrics"]["mae"], 6)),
        ("Test RMSE (N=100)", "-", round(optimal_info["test_metrics"]["rmse"], 6)),
    ]
    for r_i, r_vals in enumerate(summary_rows, 3):
        for c_i, val in enumerate(r_vals, 1):
            cell = ws_sum.cell(row=r_i, column=c_i, value=val)
            cell.border = thin_border
            if r_i == 3:
                cell.fill = header_fill
                cell.font = font_header
            else:
                cell.font = font_bold if c_i == 1 else font_reg
    for col_i in range(1, 4):
        ws_sum.column_dimensions[get_column_letter(col_i)].width = 28

    # 2. Xuất 3 sheet kết quả thực nghiệm A, B, C
    ws_a, s_a, e_a = write_dataframe_sheet(
        "Nhom_A_Quy_Mo", "THỰC NGHIỆM NHÓM A: QUY MÔ DỮ LIỆU (A)", 
        df_a, num_cols=list(range(2, 12))
    )
    ws_b, s_b, e_b = write_dataframe_sheet(
        "Nhom_B_Do_Rong", "THỰC NGHIỆM NHÓM B: ĐỘ RỘNG TẦNG ẨN (B)", 
        df_b, num_cols=list(range(2, 10))
    )
    ws_c, s_c, e_c = write_dataframe_sheet(
        "Nhom_C_Do_Sau", "THỰC NGHIỆM NHÓM C: ĐỘ SÂU MẠNG (C) & THỜI GIAN", 
        df_c, num_cols=list(range(3, 10)), time_cols=[10, 11]
    )

    wb.save(filepath)
    print(f"-> Đã lưu toàn bộ báo cáo thực nghiệm Excel tại: {filepath}")