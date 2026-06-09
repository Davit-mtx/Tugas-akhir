import os
import re
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from backend import InputValidationBackend, ProcessPipelineBackend,EvaluationMetricsBackend, HistogramGeneratorBackend,


# =========================================================
# TEMA GUI
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

COLOR_BG = "#211A2F"
COLOR_SIDEBAR = "#2B233B"
COLOR_CARD = "#332A45"
COLOR_CARD_2 = "#3B314F"
COLOR_GOLD = "#FFD000"
COLOR_GOLD_DARK = "#8A7426"
COLOR_TEXT = "#F5F1FF"
COLOR_MUTED = "#C8BEDA"
COLOR_BORDER = "#15101F"
COLOR_SUCCESS = "#2E7D32"
COLOR_ERROR = "#B00020"


class CryptoGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistem Enkripsi Citra Berbasis Chaos")
        self.geometry("1350x760")
        self.minsize(1150, 680)
        self.configure(fg_color=COLOR_BG)

        # Backend untuk input & validasi
        self.validator = InputValidationBackend()
        self.csv_result = None
        self.validation_payload = None

        # Data internal GUI
        self.optimized_csv_path = None
        self.selected_image_path = None
        self.selected_image_row = None
        self.is_image_valid = False

        self.processor = ProcessPipelineBackend(output_root="output/gui_process")
        self.process_result = None

        self.metric_evaluator = EvaluationMetricsBackend()
        self.histogram_generator = HistogramGeneratorBackend(output_root="output/gui_histogram")

        self.metrics_result = None
        self.histogram_result = None

        self.nav_buttons = {}
        self.pages = {}

        # Variabel keputusan baseline sementara
        # Nanti bagian ini bisa disesuaikan dengan X0 asli di program kamu
        self.X0 = {
            "X1": "0.1",
            "X2": "1e-6",
            "X3": "1.0",
            "...": "...",
            "Dimensi": "1 × 10⁴"
        }

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.create_header()
        self.create_sidebar()
        self.create_content_area()
        self.create_pages()

        self.show_page("Beranda")

    # =====================================================
    # FUNGSI UPDATE TEXTBOX
    # =====================================================

    def update_textbox(self, textbox, text):
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")

    def update_x0_text(self, text):
        self.update_textbox(self.x0_textbox, text)

    def update_xopt_text(self, text):
        self.update_textbox(self.xopt_textbox, text)
    # =====================================================
    # HEADER
    # =====================================================

    def create_header(self):
        self.header = ctk.CTkFrame(
            self,
            height=80,
            fg_color=COLOR_CARD,
            corner_radius=12
        )
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        self.header.grid_columnconfigure(1, weight=1)

        self.header_title = ctk.CTkLabel(
            self.header,
            text="⌂  Beranda",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLOR_GOLD
        )
        self.header_title.grid(row=0, column=0, sticky="w", padx=25, pady=20)

        subtitle = ctk.CTkLabel(
            self.header,
            text="Perbandingan Baseline dan Optimized pada Enkripsi Citra Berbasis Chaos",
            font=ctk.CTkFont(size=14),
            text_color=COLOR_MUTED
        )
        subtitle.grid(row=0, column=1, sticky="e", padx=25, pady=20)

    # =====================================================
    # SIDEBAR
    # =====================================================

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(
            self,
            width=250,
            fg_color=COLOR_SIDEBAR,
            corner_radius=12
        )
        sidebar.grid(row=1, column=0, sticky="nsw", padx=(10, 5), pady=(5, 10))
        sidebar.grid_propagate(False)

        app_label = ctk.CTkLabel(
            sidebar,
            text="🛡️  Sistem Enkripsi\nCitra Berbasis Chaos",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_GOLD,
            justify="center"
        )
        app_label.pack(padx=15, pady=(20, 25), fill="x")

        menu_items = [
            ("Beranda", "⌂"),
            ("Input & Validasi", "📁"),
            ("Proses Citra", "🔒"),
            ("Evaluasi Metrik", "📊"),
            ("Histogram", "▥"),
        ]

        for name, icon in menu_items:
            btn = ctk.CTkButton(
                sidebar,
                text=f"{icon}   {name}",
                height=44,
                anchor="w",
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color="transparent",
                text_color=COLOR_GOLD,
                hover_color=COLOR_GOLD_DARK,
                command=lambda page=name: self.show_page(page)
            )
            btn.pack(fill="x", padx=14, pady=6)
            self.nav_buttons[name] = btn

        version = ctk.CTkLabel(
            sidebar,
            text="Versi 1.0\n© 2026",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_MUTED,
            justify="left"
        )
        version.pack(side="bottom", anchor="w", padx=20, pady=20)

    # =====================================================
    # CONTENT
    # =====================================================

    def create_content_area(self):
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(5, 10))
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

    def create_pages(self):
        self.pages["Beranda"] = self.page_dashboard()
        self.pages["Input & Validasi"] = self.page_input_validation()
        self.pages["Proses Citra"] = self.page_process()
        self.pages["Evaluasi Metrik"] = self.page_metrics()
        self.pages["Histogram"] = self.page_histogram()

    def show_page(self, page_name):
        for page in self.pages.values():
            page.grid_forget()

        self.pages[page_name].grid(row=0, column=0, sticky="nsew")

        icon = self.get_page_icon(page_name)
        self.header_title.configure(text=f"{icon}  {page_name}")

        for name, btn in self.nav_buttons.items():
            if name == page_name:
                btn.configure(fg_color=COLOR_GOLD_DARK, text_color=COLOR_TEXT)
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_GOLD)

    def get_page_icon(self, page_name):
        icons = {
            "Beranda": "⌂",
            "Input & Validasi": "📁",
            "Proses Citra": "🔒",
            "Evaluasi Metrik": "📊",
            "Histogram": "▥",
        }
        return icons.get(page_name, "")

    # =====================================================
    # KOMPONEN BANTUAN
    # =====================================================

    def create_card(self, parent, title, description, icon="", row=0, column=0):
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_CARD,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_BORDER
        )
        card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)

        icon_label = ctk.CTkLabel(
            card,
            text=icon,
            font=ctk.CTkFont(size=38),
            text_color=COLOR_GOLD
        )
        icon_label.pack(pady=(18, 5))

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLOR_GOLD
        )
        title_label.pack(pady=(0, 5))

        desc_label = ctk.CTkLabel(
            card,
            text=description,
            font=ctk.CTkFont(size=13),
            text_color=COLOR_TEXT,
            wraplength=220,
            justify="center"
        )
        desc_label.pack(padx=15, pady=(0, 18))

        return card

    def create_image_panel(self, parent, title, row, column):
        panel = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=12)
        panel.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            panel,
            text=title,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_GOLD
        )
        title_label.grid(row=0, column=0, pady=(12, 5))

        img_label = ctk.CTkLabel(
            panel,
            text="Belum ada gambar",
            fg_color=COLOR_CARD_2,
            text_color=COLOR_MUTED,
            corner_radius=10
        )
        img_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(5, 12))

        return img_label

    def display_image(self, label, file_path, max_size=(330, 210)):
        image = Image.open(file_path)
        image.thumbnail(max_size)

        ctk_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size
        )

        label.configure(image=ctk_image, text="")
        label.image = ctk_image

    # =====================================================
    # PAGE 1: BERANDA
    # =====================================================

    def page_dashboard(self):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        frame.grid_rowconfigure(2, weight=1)

        hero = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        hero.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        hero.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            hero,
            text="Selamat Datang!",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=COLOR_GOLD
        )
        title.grid(row=0, column=0, sticky="w", padx=25, pady=(22, 5))

        desc = ctk.CTkLabel(
            hero,
            text=(
                "Sistem ini digunakan untuk membaca hasil optimasi parameter dari file CSV, "
                "memvalidasi citra berdasarkan database penelitian, kemudian menjalankan "
                "perbandingan proses baseline dan optimized pada skema enkripsi citra berbasis chaos."
            ),
            font=ctk.CTkFont(size=16),
            text_color=COLOR_TEXT,
            justify="left",
            wraplength=850
        )
        desc.grid(row=1, column=0, sticky="w", padx=25, pady=(0, 25))

        icon = ctk.CTkLabel(
            hero,
            text="🔐",
            font=ctk.CTkFont(size=90),
            text_color=COLOR_GOLD
        )
        icon.grid(row=0, column=1, rowspan=2, padx=40, pady=15)

        self.create_card(
            frame,
            "Input CSV Optimized",
            "Membaca file CSV hasil optimasi parameter HO.",
            "📄",
            row=1,
            column=0
        )

        self.create_card(
            frame,
            "Validasi Citra",
            "Citra dicek apakah tersedia dalam database penelitian.",
            "✅",
            row=1,
            column=1
        )

        self.create_card(
            frame,
            "Baseline vs Optimized",
            "Menjalankan dua proses menggunakan X0 dan X*(P).",
            "⚖️",
            row=1,
            column=2
        )

        self.create_card(
            frame,
            "Evaluasi Hasil",
            "Menampilkan metrik dan histogram untuk perbandingan hasil.",
            "📊",
            row=1,
            column=3
        )

        workflow = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        workflow.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=8, pady=8)
        workflow.grid_columnconfigure((0, 2, 4, 6, 8), weight=1)

        wtitle = ctk.CTkLabel(
            workflow,
            text="System Workflow",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLOR_GOLD
        )
        wtitle.grid(row=0, column=0, columnspan=9, sticky="w", padx=25, pady=(20, 10))

        steps = [
            ("📄", "Input CSV"),
            ("🖼️", "Input Citra"),
            ("✅", "Validasi"),
            ("🔒", "Baseline & Optimized"),
            ("📊", "Hasil Evaluasi"),
        ]

        for i, (icon_text, label_text) in enumerate(steps):
            col = i * 2

            box = ctk.CTkFrame(
                workflow,
                fg_color=COLOR_CARD_2,
                corner_radius=12,
                border_width=2,
                border_color=COLOR_BORDER
            )
            box.grid(row=1, column=col, sticky="ew", padx=8, pady=(5, 25))

            icon_label = ctk.CTkLabel(
                box,
                text=icon_text,
                font=ctk.CTkFont(size=34),
                text_color=COLOR_TEXT
            )
            icon_label.pack(pady=(15, 5))

            text_label = ctk.CTkLabel(
                box,
                text=label_text,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLOR_TEXT
            )
            text_label.pack(pady=(0, 15))

            if i < len(steps) - 1:
                arrow = ctk.CTkLabel(
                    workflow,
                    text="→",
                    font=ctk.CTkFont(size=32, weight="bold"),
                    text_color=COLOR_GOLD
                )
                arrow.grid(row=1, column=col + 1, padx=0, pady=(5, 25))

        return frame

    # =====================================================
    # PAGE 2: INPUT & VALIDASI
    # =====================================================

    def page_input_validation(self):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # Panel input
        input_panel = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        input_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        input_panel.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            input_panel,
            text="Input File dan Validasi Citra",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_GOLD
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(18, 12))

        csv_btn = ctk.CTkButton(
            input_panel,
            text="📄 Input CSV Optimized",
            fg_color=COLOR_GOLD,
            text_color="#1B1525",
            hover_color="#E0B900",
            font=ctk.CTkFont(weight="bold"),
            command=self.load_optimized_csv
        )
        csv_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.csv_status_label = ctk.CTkLabel(
            input_panel,
            text="Belum ada CSV dipilih",
            text_color=COLOR_MUTED,
            anchor="w"
        )
        self.csv_status_label.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        img_btn = ctk.CTkButton(
            input_panel,
            text="🖼️ Input Citra",
            fg_color="transparent",
            border_width=2,
            border_color=COLOR_GOLD,
            text_color=COLOR_GOLD,
            hover_color=COLOR_GOLD_DARK,
            font=ctk.CTkFont(weight="bold"),
            command=self.load_and_validate_image
        )
        img_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.image_status_label = ctk.CTkLabel(
            input_panel,
            text="Belum ada citra dipilih",
            text_color=COLOR_MUTED,
            anchor="w"
        )
        self.image_status_label.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.validation_label = ctk.CTkLabel(
            input_panel,
            text="Status validasi: menunggu input",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_MUTED
        )
        self.validation_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=20, pady=(10, 20))

        # Preview citra
        preview_panel = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        preview_panel.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        preview_panel.grid_rowconfigure(1, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)

        ptitle = ctk.CTkLabel(
            preview_panel,
            text="Preview Citra Valid",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_GOLD
        )
        ptitle.grid(row=0, column=0, sticky="w", padx=20, pady=(18, 12))

        self.valid_image_preview = ctk.CTkLabel(
            preview_panel,
            text="Citra valid akan ditampilkan di sini",
            fg_color=COLOR_CARD_2,
            text_color=COLOR_MUTED,
            corner_radius=10
        )
        self.valid_image_preview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # Variabel keputusan
        var_panel = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        var_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        var_panel.grid_columnconfigure(0, weight=1)
        var_panel.grid_columnconfigure(1, weight=1)

        xtitle = ctk.CTkLabel(
            var_panel,
            text="Variabel Keputusan yang Digunakan",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_GOLD
        )
        xtitle.grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(18, 10))

        self.x0_textbox = ctk.CTkTextbox(
            var_panel,
            height=180,
            fg_color=COLOR_CARD_2,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(size=14, family="Consolas")
        )
        self.x0_textbox.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        self.x0_textbox.insert("1.0", self.format_x0())
        self.x0_textbox.configure(state="disabled")

        self.xopt_textbox = ctk.CTkTextbox(
            var_panel,
            height=180,
            fg_color=COLOR_CARD_2,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(size=14, family="Consolas")
        )
        self.xopt_textbox.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        self.xopt_textbox.insert("1.0", "X*(P) Optimized\n\nMenunggu CSV dan citra valid.")
        self.xopt_textbox.configure(state="disabled")

        return frame

    def format_x0(self):
        x0 = self.validator.get_baseline_x0()
        return self.validator.format_parameter_text("X0 Baseline", x0)

    # =====================================================
    # FUNGSI INPUT CSV DAN VALIDASI CITRA
    # =====================================================

    def load_optimized_csv(self):
        file_path = filedialog.askopenfilename(
            title="Pilih CSV Hasil Optimized",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        result = self.validator.load_csv(file_path)
        self.csv_result = result

        if result["ok"]:
            self.csv_status_label.configure(
                text=f"CSV valid: {result['file_name']} | {result['n_rows']} data",
                text_color=COLOR_TEXT
            )

            category_counts = result.get("category_counts", {})
            category_info = ", ".join(
                [f"{key}: {value}" for key, value in category_counts.items()]
            )

            messagebox.showinfo(
                "CSV Berhasil Dibaca",
                (
                    f"{result['message']}\n\n"
                    f"Jumlah data: {result['n_rows']}\n"
                    f"Jumlah kolom: {result['n_columns']}\n"
                    f"Kategori: {category_info}"
                )
            )

            # Reset status citra setelah CSV baru dimasukkan
            self.is_image_valid = False
            self.selected_image_path = None
            self.validation_payload = None

            self.image_status_label.configure(
                text="Belum ada citra dipilih",
                text_color=COLOR_MUTED
            )
            self.validation_label.configure(
                text="Status validasi: menunggu input citra",
                text_color=COLOR_MUTED
            )

            self.valid_image_preview.configure(
                image=None,
                text="Citra valid akan ditampilkan di sini"
            )

            self.update_x0_text(self.format_x0())
            self.update_xopt_text("X*(P_i) Optimized\n\nMenunggu citra valid.")

        else:
            self.csv_status_label.configure(
                text="CSV tidak valid",
                text_color=COLOR_ERROR
            )

            missing = result.get("missing_columns", [])
            missing_text = "\n".join(missing) if missing else "-"

            messagebox.showerror(
                "CSV Tidak Valid",
                (
                    f"{result['message']}\n\n"
                    f"Kolom yang hilang:\n{missing_text}"
                )
            )

    def load_and_validate_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Citra",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        result = self.validator.validate_image(file_path)
        self.validation_payload = result

        # DEBUG SEMENTARA
        print("HASIL VALIDASI IMAGE:")
        print(result)

        if result["ok"]:
            self.is_image_valid = True
            self.selected_image_path = result["image_path"]

            self.image_status_label.configure(
                text=f"Diterima: {result['input_file_name']}",
                text_color=COLOR_TEXT
            )

            self.validation_label.configure(
                text=(
                    f"Status validasi: CITRA VALID | "
                    f"Kategori: {result['category']} | "
                    f"Best Fitness: {result['best_fitness']}"
                ),
                text_color=COLOR_SUCCESS
            )

            self.display_image(
                self.valid_image_preview,
                result["image_path"],
                max_size=(420, 260)
            )

            x0_text = self.validator.format_parameter_text(
                "X0 Baseline",
                result["x0"]
            )

            xopt_text = self.validator.format_parameter_text(
                "X*(P_i) Optimized",
                result["xopt"]
            )

            summary_text = self.validator.format_validation_summary_text(result)

            self.update_x0_text(x0_text)
            self.update_xopt_text(
                xopt_text + "\n\n" + summary_text
            )

            messagebox.showinfo(
                "Citra Valid",
                result["message"]
            )

        else:
            self.is_image_valid = False
            self.selected_image_path = None

            self.image_status_label.configure(
                text=f"Ditolak: {result['input_file_name']}",
                text_color=COLOR_ERROR
            )

            self.validation_label.configure(
                text="Status validasi: CITRA TIDAK VALID",
                text_color=COLOR_ERROR
            )

            self.valid_image_preview.configure(
                image=None,
                text="Citra ditolak.\nSilakan input citra lain."
            )

            self.update_xopt_text(
                "X*(P_i) Optimized\n\n" + result["message"]
            )

            messagebox.showerror(
                "Citra Ditolak",
                result["message"]
            )
    # =====================================================
    # PAGE 3: PROSES CITRA
    # =====================================================

    def page_process(self):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid_columnconfigure((0, 1, 2), weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        control = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        control.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        control.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            control,
            text="Proses Baseline dan Optimized",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_GOLD
        )
        title.grid(row=0, column=0, sticky="w", padx=20, pady=(16, 5))

        desc = ctk.CTkLabel(
            control,
            text=(
                "Program akan menjalankan dua skenario: baseline menggunakan X0 dan optimized "
                "menggunakan X*(P) yang dibaca dari CSV hasil optimasi."
            ),
            font=ctk.CTkFont(size=14),
            text_color=COLOR_TEXT,
            wraplength=850,
            justify="left"
        )
        desc.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 16))

        run_btn = ctk.CTkButton(
            control,
            text="▶ Jalankan Baseline & Optimized",
            fg_color=COLOR_GOLD,
            text_color="#1B1525",
            hover_color="#E0B900",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.run_pipeline_placeholder
        )
        run_btn.grid(row=0, column=1, rowspan=2, sticky="e", padx=20, pady=16)

        self.original_process_view = self.create_image_panel(frame, "Citra Asli / Plaintext", 1, 0)
        self.baseline_cipher_view = self.create_image_panel(frame, "Baseline - Cipher Image", 1, 1)
        self.baseline_decrypt_view = self.create_image_panel(frame, "Baseline - Citra Dekripsi", 1, 2)

        self.optimized_plain_view = self.create_image_panel(frame, "Citra Valid", 2, 0)
        self.optimized_cipher_view = self.create_image_panel(frame, "Optimized - Cipher Image", 2, 1)
        self.optimized_decrypt_view = self.create_image_panel(frame, "Optimized - Citra Dekripsi", 2, 2)

        return frame

    def run_pipeline_placeholder(self):
        payload = self.validator.get_current_payload()

        if not payload.get("ok"):
            messagebox.showwarning(
                "Citra Belum Valid",
                "Masukkan CSV optimized dan citra yang valid terlebih dahulu pada tab Input & Validasi."
            )
            return

        result = self.processor.run_baseline_and_optimized(payload)
        self.process_result = result

        if not result["ok"]:
            messagebox.showerror(
                "Proses Gagal",
                result["message"]
            )
            return

        # Tampilkan citra asli
        self.display_image(
            self.original_process_view,
            result["input"]["original_path"]
        )

        self.display_image(
            self.optimized_plain_view,
            result["input"]["original_path"]
        )

        # Tampilkan hasil baseline
        self.display_image(
            self.baseline_cipher_view,
            result["baseline"]["cipher_path"]
        )

        self.display_image(
            self.baseline_decrypt_view,
            result["baseline"]["decrypted_path"]
        )

        # Tampilkan hasil optimized
        self.display_image(
            self.optimized_cipher_view,
            result["optimized"]["cipher_path"]
        )

        self.display_image(
            self.optimized_decrypt_view,
            result["optimized"]["decrypted_path"]
        )

        messagebox.showinfo(
            "Proses Selesai",
            (
                "Baseline dan optimized berhasil dijalankan.\n\n"
                f"Baseline Lossless  : {result['baseline']['lossless']}\n"
                f"Optimized Lossless : {result['optimized']['lossless']}\n\n"
                f"Baseline Enc Time  : {result['baseline']['enc_time']} s\n"
                f"Optimized Enc Time : {result['optimized']['enc_time']} s"
            )
        )

    # =====================================================
    # PAGE 4: EVALUASI METRIK
    # =====================================================

    def page_metrics(self):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        baseline_panel = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        baseline_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        opt_panel = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=12)
        opt_panel.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        self.create_metric_table_placeholder(
            baseline_panel,
            "Evaluasi Metrik - Baseline"
        )

        self.create_metric_table_placeholder(
            opt_panel,
            "Evaluasi Metrik - Optimized"
        )

        return frame

    def create_metric_table_placeholder(self, parent, title):
        title_label = ctk.CTkLabel(
            parent,
            text=title,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_GOLD
        )
        title_label.pack(anchor="w", padx=20, pady=(20, 10))

        table_text = (
            "Metrik              R          G          B       Average\n"
            "----------------------------------------------------------\n"
            "Entropy             -          -          -          -\n"
            "Corr Horizontal     -          -          -          -\n"
            "Corr Vertical       -          -          -          -\n"
            "Corr Diagonal       -          -          -          -\n"
            "NPCR (%)            -          -          -          -\n"
            "UACI (%)            -          -          -          -\n"
            "Waktu Enkripsi      -          -          -          -\n"
            "Waktu Dekripsi      -          -          -          -\n"
            "Lossless            -          -          -          -\n"
        )

        textbox = ctk.CTkTextbox(
            parent,
            fg_color=COLOR_CARD_2,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(size=14, family="Consolas")
        )
        textbox.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        textbox.insert("1.0", table_text)
        textbox.configure(state="disabled")

        return textbox

    # =====================================================
    # PAGE 5: HISTOGRAM
    # =====================================================

    def page_histogram(self):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        self.create_histogram_panel(frame, "Baseline - Plaintext", 0, 0)
        self.create_histogram_panel(frame, "Baseline - Cipher Image", 0, 1)
        self.create_histogram_panel(frame, "Optimized - Plaintext", 1, 0)
        self.create_histogram_panel(frame, "Optimized - Cipher Image", 1, 1)

        return frame

    def create_histogram_panel(self, parent, title, row, column):
        panel = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=12)
        panel.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            panel,
            text=title,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLOR_GOLD
        )
        title_label.grid(row=0, column=0, sticky="w", padx=20, pady=(16, 8))

        placeholder = ctk.CTkLabel(
            panel,
            text="Histogram akan ditampilkan di sini.",
            fg_color=COLOR_CARD_2,
            text_color=COLOR_MUTED,
            corner_radius=10
        )
        placeholder.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))


if __name__ == "__main__":
    app = CryptoGUI()
    app.mainloop()