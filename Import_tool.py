import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import subprocess
import sys


class FileImportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 E57 Point Cloud Pipeline Tool")
        self.root.geometry("650x600")
        self.root.resizable(False, False)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Variables
        self.file_path = ""
        self.excel_output_path = ""
        self.is_processing = False
        self.start_time = 0
        self.pipeline_process = None
        self.output_log = []

        # Load icons
        self.load_icons()

        # Create the UI
        self.create_widgets()

    def load_icons(self):
        """Load Unicode symbol icons"""
        self.icons = {
            'import': '📥', 'select': '📂', 'cancel': '❌', 'success': '✅',
            'error': '❌', 'warning': '⚠️', 'info': 'ℹ️', 'processing': '🔄',
            'file': '📄', 'rocket': '🚀', 'chart': '📊', 'start': '▶️',
            'stop': '⏹️', 'pipeline': '🔧', 'e57': '📐', 'excel': '📊',
            'folder': '📁', 'debug': '🐛', 'log': '📝'
        }

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="25")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 25))

        title_label = ttk.Label(title_frame, text="🚀 E57 Point Cloud Pipeline",
                                font=("Segoe UI", 18, "bold"))
        title_label.pack()

        subtitle_label = ttk.Label(title_frame, text="Process E57 files with debugging and file validation",
                                   font=("Segoe UI", 10), foreground="gray")
        subtitle_label.pack(pady=(5, 0))

        # Input file selection
        input_section = ttk.LabelFrame(main_frame, text="📐 E57 Input File", padding="15")
        input_section.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        input_section.grid_columnconfigure(1, weight=1)

        self.input_btn = ttk.Button(input_section,
                                    text=f"{self.icons['e57']} Select E57 File",
                                    command=self.select_input_file)
        self.input_btn.grid(row=0, column=0, padx=(0, 15), sticky=tk.W)

        self.input_label = ttk.Label(input_section, text="No E57 file selected",
                                     foreground="gray", font=("Segoe UI", 9))
        self.input_label.grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Output file selection
        output_section = ttk.LabelFrame(main_frame, text="📊 Excel Output File", padding="15")
        output_section.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        output_section.grid_columnconfigure(1, weight=1)

        self.output_btn = ttk.Button(output_section,
                                     text=f"{self.icons['excel']} Set Output Location",
                                     command=self.select_output_file)
        self.output_btn.grid(row=0, column=0, padx=(0, 15), sticky=tk.W)

        self.output_label = ttk.Label(output_section, text="No output location set",
                                      foreground="gray", font=("Segoe UI", 9))
        self.output_label.grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Quick action buttons
        quick_section = ttk.Frame(output_section)
        quick_section.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))

        self.folder_btn = ttk.Button(quick_section,
                                     text=f"{self.icons['folder']} Open Output Folder",
                                     command=self.open_output_folder,
                                     state="disabled")
        self.folder_btn.grid(row=0, column=0, sticky=tk.W)

        # Action buttons section
        button_section = ttk.LabelFrame(main_frame, text="🔧 Pipeline Control", padding="15")
        button_section.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        button_section.grid_columnconfigure(0, weight=1)
        button_section.grid_columnconfigure(1, weight=1)
        button_section.grid_columnconfigure(2, weight=1)

        # START PIPELINE BUTTON
        self.start_btn = ttk.Button(button_section,
                                    text=f"{self.icons['start']} Start Pipeline",
                                    command=self.start_pipeline,
                                    state="disabled")
        self.start_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        # Cancel button
        self.cancel_btn = ttk.Button(button_section,
                                     text=f"{self.icons['cancel']} Cancel",
                                     command=self.cancel_process,
                                     state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))

        # Debug button
        self.debug_btn = ttk.Button(button_section,
                                    text=f"{self.icons['debug']} Show Log",
                                    command=self.show_debug_log)
        self.debug_btn.grid(row=0, column=2, padx=(5, 0), sticky=(tk.W, tk.E))

        # Progress section
        progress_section = ttk.LabelFrame(main_frame, text="📊 Processing Progress", padding="15")
        progress_section.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        progress_section.grid_columnconfigure(0, weight=1)

        # Pipeline Progress bar
        pipeline_label = ttk.Label(progress_section, text="E57 Pipeline Progress:", font=("Segoe UI", 9, "bold"))
        pipeline_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.pipeline_progress = ttk.Progressbar(progress_section, mode='indeterminate', length=500)
        self.pipeline_progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Progress info frame
        progress_info = ttk.Frame(progress_section)
        progress_info.grid(row=2, column=0, sticky=(tk.W, tk.E))
        progress_info.grid_columnconfigure(1, weight=1)

        # Status section
        status_section = ttk.LabelFrame(main_frame, text="📋 Status & Debug Info", padding="15")
        status_section.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        status_section.grid_columnconfigure(0, weight=1)

        # Status label
        self.status_label = ttk.Label(status_section,
                                      text=f"{self.icons['info']} Ready - Please select input and output files",
                                      font=("Segoe UI", 10), foreground="blue")
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Command display
        self.command_label = ttk.Label(status_section, text="",
                                       font=("Courier", 8), foreground="gray")
        self.command_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

    def select_input_file(self):
        """Select E57 input file"""
        file_path = filedialog.askopenfilename(
            title="Select E57 input file",
            filetypes=[
                ("E57 files", "*.e57"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            file_size = self.format_size(os.path.getsize(file_path))

            self.input_label.config(
                text=f"{self.icons['e57']} {filename} ({file_size})",
                foreground="green"
            )

            # Auto-suggest output filename
            if not self.excel_output_path:
                base_name = os.path.splitext(filename)[0]
                suggested_output = os.path.join(os.path.dirname(file_path), f"{base_name}_output.xlsx")
                self.excel_output_path = suggested_output
                self.output_label.config(
                    text=f"{self.icons['excel']} {os.path.basename(suggested_output)} (auto-suggested)",
                    foreground="orange"
                )
                self.folder_btn.config(state="normal")

            self.check_ready_state()

    def select_output_file(self):
        """Select Excel output file location"""
        # Suggest initial directory and filename
        initial_dir = os.path.dirname(self.file_path) if self.file_path else os.getcwd()
        initial_name = ""
        if self.file_path:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            initial_name = f"{base_name}_output.xlsx"

        file_path = filedialog.asksaveasfilename(
            title="Set Excel output file location",
            initialdir=initial_dir,
            initialfile=initial_name,
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("Excel files", "*.xls"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.excel_output_path = file_path
            filename = os.path.basename(file_path)

            self.output_label.config(
                text=f"{self.icons['excel']} {filename}",
                foreground="green"
            )
            self.folder_btn.config(state="normal")
            self.check_ready_state()

    def open_output_folder(self):
        """Open the folder containing the output file"""
        if self.excel_output_path:
            folder_path = os.path.dirname(self.excel_output_path)
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(folder_path)
                elif os.name == 'posix':  # macOS and Linux
                    subprocess.call(['open', folder_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {e}")

    def check_ready_state(self):
        """Check if both input and output are selected"""
        if self.file_path and self.excel_output_path:
            self.start_btn.config(state="normal")
            self.status_label.config(
                text=f"{self.icons['success']} Ready to start E57 pipeline processing",
                foreground="green"
            )
        else:
            missing = []
            if not self.file_path:
                missing.append("input E57 file")
            if not self.excel_output_path:
                missing.append("output Excel file")

            self.status_label.config(
                text=f"{self.icons['warning']} Please select: {', '.join(missing)}",
                foreground="orange"
            )

    def start_pipeline(self):
        """Start the E57 point cloud pipeline process"""
        if not self.file_path or not self.excel_output_path:
            messagebox.showerror("❌ Error", "Please select both input E57 file and output Excel file!")
            return

        # Check if pipeline file exists
        pipeline_script = "e57_pointcloud_pipeline_v1.py"
        if not os.path.exists(pipeline_script):
            messagebox.showerror("❌ Error",
                                 f"Pipeline script '{pipeline_script}' not found in current directory!\n\nCurrent directory: {os.getcwd()}")
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(self.excel_output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.add_log(f"Created output directory: {output_dir}")
            except Exception as e:
                messagebox.showerror("❌ Error", f"Could not create output directory: {e}")
                return

        self.is_processing = True
        self.start_btn.config(state="disabled")
        self.input_btn.config(state="disabled")
        self.output_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")

        # Clear previous logs
        self.output_log = []

        # Start progress animation
        self.pipeline_progress.config(mode='indeterminate')
        self.pipeline_progress.start(10)

        self.status_label.config(
            text=f"{self.icons['pipeline']} Starting E57 pipeline processing...",
            foreground="purple"
        )

        # Start pipeline in a separate thread
        pipeline_thread = threading.Thread(target=self.run_pipeline)
        pipeline_thread.daemon = True
        pipeline_thread.start()

    def add_log(self, message):
        """Add message to debug log"""
        timestamp = time.strftime("%H:%M:%S")
        self.output_log.append(f"[{timestamp}] {message}")

    def run_pipeline(self):
        """Run the external E57 pipeline script with proper arguments"""
        try:
            self.start_time = time.time()

            # Construct command with required arguments
            cmd = [
                sys.executable,
                "e57_pointcloud_pipeline_v1.py",
                "--input", self.file_path,
                "--excel_out", self.excel_output_path
            ]

            self.add_log(f"Starting pipeline with command: {' '.join(cmd)}")
            self.add_log(f"Input file: {self.file_path}")
            self.add_log(f"Output file: {self.excel_output_path}")
            self.add_log(f"Current working directory: {os.getcwd()}")

            # Update command display in UI
            self.root.after(0, lambda: self.command_label.config(text=f"Command: {' '.join(cmd)}"))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()  # Explicitly set working directory
            )

            self.pipeline_process = process

            # Wait for process to complete and capture all output
            stdout, stderr = process.communicate()

            self.add_log(f"Process completed with return code: {process.returncode}")

            if stdout:
                self.add_log("STDOUT:")
                for line in stdout.splitlines():
                    self.add_log(f"  {line}")

            if stderr:
                self.add_log("STDERR:")
                for line in stderr.splitlines():
                    self.add_log(f"  {line}")

            if self.is_processing:
                if process.returncode == 0:
                    # Check if output file was actually created
                    if os.path.exists(self.excel_output_path):
                        file_size = os.path.getsize(self.excel_output_path)
                        self.add_log(f"Output file created successfully: {self.excel_output_path} ({file_size} bytes)")
                        self.root.after(0, self.pipeline_complete)
                    else:
                        self.add_log(
                            f"WARNING: Pipeline completed but output file not found at: {self.excel_output_path}")
                        self.root.after(0, lambda: self.pipeline_error(
                            "Pipeline completed but output file was not created"))
                else:
                    error_msg = stderr if stderr else f"Process failed with return code {process.returncode}"
                    self.root.after(0, lambda: self.pipeline_error(error_msg))

        except Exception as e:
            self.add_log(f"Exception occurred: {str(e)}")
            self.root.after(0, lambda: self.pipeline_error(str(e)))

    def pipeline_complete(self):
        """Handle successful pipeline completion"""
        self.pipeline_progress.stop()
        self.pipeline_progress.config(mode='determinate', value=100)

        elapsed_time = time.time() - self.start_time
        self.status_label.config(
            text=f"{self.icons['success']} Pipeline completed in {elapsed_time:.1f}s! Output file created.",
            foreground="green"
        )
        self.reset_ui()

        # Show success message with option to open file/folder
        result = messagebox.askyesno(
            "🎉 Success",
            f"E57 Point Cloud Pipeline completed successfully!\n\nOutput saved to:\n{self.excel_output_path}\n\nWould you like to open the output folder?"
        )

        if result:
            self.open_output_folder()

    def pipeline_error(self, error_msg):
        """Handle pipeline error"""
        self.pipeline_progress.stop()
        self.pipeline_progress.config(value=0)

        self.status_label.config(
            text=f"{self.icons['error']} Pipeline failed - check debug log for details",
            foreground="red"
        )
        self.reset_ui()

        # Show error with option to view debug log
        result = messagebox.askyesnocancel(
            "❌ Pipeline Error",
            f"Pipeline processing failed:\n\n{error_msg[:200]}{'...' if len(error_msg) > 200 else ''}\n\nWould you like to see the detailed debug log?"
        )

        if result:
            self.show_debug_log()

    def show_debug_log(self):
        """Show debug log in a new window"""
        log_window = tk.Toplevel(self.root)
        log_window.title("🐛 Debug Log")
        log_window.geometry("800x500")

        # Create text widget with scrollbar
        frame = ttk.Frame(log_window, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        log_window.grid_columnconfigure(0, weight=1)
        log_window.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        text_widget = tk.Text(frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Insert log content
        log_content = "\n".join(self.output_log) if self.output_log else "No log entries available."
        text_widget.insert("1.0", log_content)
        text_widget.config(state="disabled")

        # Close button
        close_btn = ttk.Button(frame, text="Close", command=log_window.destroy)
        close_btn.grid(row=1, column=0, pady=(10, 0))

    def cancel_process(self):
        """Cancel the ongoing process"""
        if self.is_processing:
            self.is_processing = False
            if self.pipeline_process:
                self.pipeline_process.terminate()
                self.add_log("Process terminated by user")

            self.pipeline_progress.stop()
            self.pipeline_progress.config(value=0)
            self.status_label.config(
                text=f"{self.icons['warning']} Pipeline process cancelled by user",
                foreground="red"
            )
            self.reset_ui()

    def reset_ui(self):
        """Reset UI to initial state"""
        self.start_btn.config(state="normal" if (self.file_path and self.excel_output_path) else "disabled")
        self.input_btn.config(state="normal")
        self.output_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.is_processing = False

    def format_size(self, size):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FileImportApp(root)
    root.mainloop()
