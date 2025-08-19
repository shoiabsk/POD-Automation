import sys
import os

# Check tkinter availability first
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    print("✅ tkinter imported successfully")
except ImportError as e:
    print(f"❌ Error importing tkinter: {e}")
    print("Please install tkinter: pip install tk")
    sys.exit(1)

import threading
import time
import subprocess
import queue
import re


class FileImportApp:
    def __init__(self, root):
        print("🚀 Initializing FileImportApp...")
        self.root = root
        self.root.title("🚀 E57 Point Cloud Pipeline Tool")
        self.root.geometry("650x450")
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
        self.progress_queue = queue.Queue()
        self.current_progress = 0

        # Load icons
        self.load_icons()

        # Create the UI
        self.create_widgets()
        print("✅ FileImportApp initialized successfully")

    def load_icons(self):
        """Load Unicode symbol icons"""
        self.icons = {
            'import': '📥', 'select': '📂', 'cancel': '❌', 'success': '✅',
            'error': '❌', 'warning': '⚠️', 'info': 'ℹ️', 'processing': '🔄',
            'file': '📄', 'rocket': '🚀', 'chart': '📊', 'start': '▶️',
            'stop': '⏹️', 'pipeline': '🔧', 'e57': '📐', 'excel': '📊',
            'folder': '📁', 'clock': '⏰'
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
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        title_label = ttk.Label(title_frame, text="🚀 E57 Point Cloud Pipeline",
                                font=("Segoe UI", 18, "bold"))
        title_label.pack()

        subtitle_label = ttk.Label(title_frame, text="Execute e57_pointcloud_pipeline_v1.py with real-time monitoring",
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

        # Quick action button
        self.folder_btn = ttk.Button(output_section,
                                     text=f"{self.icons['folder']} Open Folder",
                                     command=self.open_output_folder,
                                     state="disabled")
        self.folder_btn.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)

        # Action buttons section
        button_section = ttk.LabelFrame(main_frame, text="🔧 Pipeline Control", padding="15")
        button_section.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        button_section.grid_columnconfigure(0, weight=1)
        button_section.grid_columnconfigure(1, weight=1)

        # START PIPELINE BUTTON
        self.start_btn = ttk.Button(button_section,
                                    text=f"{self.icons['start']} Run e57_pointcloud_pipeline_v1.py",
                                    command=self.start_pipeline,
                                    state="disabled")
        self.start_btn.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))

        # Cancel button
        self.cancel_btn = ttk.Button(button_section,
                                     text=f"{self.icons['cancel']} Stop Pipeline",
                                     command=self.cancel_process,
                                     state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=(10, 0), sticky=(tk.W, tk.E))

        # Progress section - LEFT/RIGHT LAYOUT
        progress_section = ttk.LabelFrame(main_frame, text="📊 Pipeline Execution Progress", padding="15")
        progress_section.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        progress_section.grid_columnconfigure(0, weight=3)
        progress_section.grid_columnconfigure(1, weight=1)

        # LEFT SIDE - Progress Bar
        left_frame = ttk.Frame(progress_section)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 20))
        left_frame.grid_columnconfigure(0, weight=1)

        pipeline_label = ttk.Label(left_frame, text="e57_pointcloud_pipeline_v1.py Execution:",
                                   font=("Segoe UI", 9, "bold"))
        pipeline_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        # Progress bar
        self.pipeline_progress = ttk.Progressbar(left_frame, mode='determinate',
                                                 length=350, maximum=100)
        self.pipeline_progress.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Processing details below progress bar
        self.process_details_label = ttk.Label(left_frame, text="",
                                               font=("Segoe UI", 8), foreground="blue")
        self.process_details_label.grid(row=2, column=0, sticky=tk.W, pady=(8, 0))

        # RIGHT SIDE - Percentage and Time
        right_frame = ttk.Frame(progress_section)
        right_frame.grid(row=0, column=1, sticky=(tk.E, tk.N), padx=(20, 0))

        # Large percentage display
        self.progress_percent_label = ttk.Label(right_frame, text="0%",
                                                font=("Segoe UI", 20, "bold"),
                                                foreground="blue")
        self.progress_percent_label.grid(row=0, column=0, sticky=tk.E)

        # Estimated time remaining
        self.estimated_time_label = ttk.Label(right_frame,
                                              text=f"{self.icons['clock']} Estimated: --",
                                              font=("Segoe UI", 10), foreground="gray")
        self.estimated_time_label.grid(row=1, column=0, sticky=tk.E, pady=(5, 0))

        # Completion time estimate
        self.completion_time_label = ttk.Label(right_frame, text="",
                                               font=("Segoe UI", 9), foreground="green")
        self.completion_time_label.grid(row=2, column=0, sticky=tk.E, pady=(5, 0))

        # Status section
        status_section = ttk.LabelFrame(main_frame, text="📋 Pipeline Status", padding="15")
        status_section.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        status_section.grid_columnconfigure(0, weight=1)

        # Status label
        self.status_label = ttk.Label(status_section,
                                      text=f"{self.icons['info']} Ready to execute e57_pointcloud_pipeline_v1.py",
                                      font=("Segoe UI", 10), foreground="blue")
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

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
                text=f"{self.icons['success']} Ready to execute e57_pointcloud_pipeline_v1.py",
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
        """Start the actual e57_pointcloud_pipeline_v1.py script"""
        if not self.file_path or not self.excel_output_path:
            messagebox.showerror("❌ Error", "Please select both input E57 file and output Excel file!")
            return

        # Check if the actual pipeline script exists
        pipeline_script = "e57_pointcloud_pipeline_v1.py"
        if not os.path.exists(pipeline_script):
            messagebox.showerror(
                "❌ Pipeline Script Not Found",
                f"Cannot find e57_pointcloud_pipeline_v1.py in current directory:\n{os.getcwd()}"
            )
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(self.excel_output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("❌ Error", f"Could not create output directory: {e}")
                return

        # Disable UI controls
        self.is_processing = True
        self.start_btn.config(state="disabled")
        self.input_btn.config(state="disabled")
        self.output_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")

        # Reset progress
        self.current_progress = 0
        self.pipeline_progress['value'] = 0
        self.progress_percent_label.config(text="0%", foreground="blue")
        self.estimated_time_label.config(text=f"{self.icons['clock']} Starting...")
        self.completion_time_label.config(text="")
        self.process_details_label.config(text="Initializing...")

        self.status_label.config(
            text=f"{self.icons['processing']} Executing e57_pointcloud_pipeline_v1.py...",
            foreground="purple"
        )

        # Start the actual pipeline process in background
        pipeline_thread = threading.Thread(target=self.run_actual_pipeline, args=(pipeline_script,))
        pipeline_thread.daemon = True
        pipeline_thread.start()

        # Start monitoring progress
        self.monitor_progress()

    def run_actual_pipeline(self, pipeline_script):
        """Run the actual e57_pointcloud_pipeline_v1.py script"""
        try:
            self.start_time = time.time()

            # Construct command to run the actual pipeline
            cmd = [
                sys.executable,
                pipeline_script,
                "--input", self.file_path,
                "--excel_out", self.excel_output_path
            ]

            print(f"🚀 Running command: {' '.join(cmd)}")

            # Execute the pipeline script
            self.pipeline_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                bufsize=1,
                universal_newlines=True
            )

            # Wait for process to complete
            stdout, stderr = self.pipeline_process.communicate()

            print(f"✅ Pipeline completed with return code: {self.pipeline_process.returncode}")

            if self.is_processing:
                if self.pipeline_process.returncode == 0:
                    # Success - check if output file exists
                    if os.path.exists(self.excel_output_path):
                        file_size = os.path.getsize(self.excel_output_path)
                        self.progress_queue.put(('complete', f"Success! Output file: {file_size} bytes"))
                    else:
                        self.progress_queue.put(('error', "Pipeline completed but output file not found"))
                else:
                    # Error occurred
                    error_msg = stderr if stderr else f"Process failed with return code {self.pipeline_process.returncode}"
                    self.progress_queue.put(('error', error_msg))

        except Exception as e:
            print(f"❌ Exception in pipeline execution: {e}")
            if self.is_processing:
                self.progress_queue.put(('error', f"Exception: {str(e)}"))

    def monitor_progress(self):
        """Monitor progress updates from the pipeline process"""
        try:
            # Check for messages from pipeline
            while True:
                try:
                    msg_type, data = self.progress_queue.get_nowait()

                    if msg_type == 'complete':
                        self.pipeline_complete(data)
                        return
                    elif msg_type == 'error':
                        self.pipeline_error(data)
                        return

                except queue.Empty:
                    break
        except:
            pass

        # Simulate progress if process is still running
        if self.is_processing:
            if hasattr(self, 'start_time'):
                elapsed = time.time() - self.start_time
                # Simulate progress based on elapsed time
                if self.current_progress < 95:
                    estimated_progress = min(95, elapsed * 1.2)  # 1.2% per second
                    if estimated_progress > self.current_progress:
                        self.update_progress(estimated_progress)

            # Continue monitoring
            self.root.after(500, self.monitor_progress)

    def update_progress(self, progress_value):
        """Update progress display"""
        self.current_progress = max(self.current_progress, progress_value)
        self.pipeline_progress['value'] = self.current_progress
        self.progress_percent_label.config(text=f"{self.current_progress:.1f}%")

        # Update time estimation
        if hasattr(self, 'start_time') and self.current_progress > 0:
            elapsed_time = time.time() - self.start_time
            total_estimated_time = elapsed_time * (100 / self.current_progress)
            remaining_time = total_estimated_time - elapsed_time

            if remaining_time > 60:
                time_str = f"{int(remaining_time // 60)}m {int(remaining_time % 60)}s"
            else:
                time_str = f"{int(remaining_time)}s"

            self.estimated_time_label.config(text=f"{self.icons['clock']} Remaining: {time_str}")

            # Calculate estimated completion time
            completion_time = time.time() + remaining_time
            completion_str = time.strftime("%I:%M %p", time.localtime(completion_time))
            self.completion_time_label.config(text=f"Complete by: {completion_str}")

        # Update process details
        self.process_details_label.config(text=f"Processing... {self.current_progress:.1f}% complete")

        # Update status
        self.status_label.config(
            text=f"{self.icons['processing']} e57_pointcloud_pipeline_v1.py running... {self.current_progress:.1f}%",
            foreground="purple"
        )

    def pipeline_complete(self, message):
        """Handle pipeline completion"""
        self.current_progress = 100
        self.pipeline_progress['value'] = 100
        self.progress_percent_label.config(text="100%", foreground="green")
        self.estimated_time_label.config(text=f"{self.icons['success']} Completed!")

        elapsed_time = time.time() - self.start_time
        completion_str = time.strftime("%I:%M %p", time.localtime())
        self.completion_time_label.config(text=f"Finished at: {completion_str}")
        self.process_details_label.config(text="Processing completed successfully!")

        self.status_label.config(
            text=f"{self.icons['success']} e57_pointcloud_pipeline_v1.py completed successfully!",
            foreground="green"
        )

        self.reset_ui()

        result = messagebox.askyesno(
            "🎉 Pipeline Complete",
            f"e57_pointcloud_pipeline_v1.py completed successfully!\n\nElapsed time: {elapsed_time:.1f}s\nOutput: {self.excel_output_path}\n\nOpen output folder?"
        )

        if result:
            self.open_output_folder()

    def pipeline_error(self, error_msg):
        """Handle pipeline error"""
        self.progress_percent_label.config(text="Error", foreground="red")
        self.estimated_time_label.config(text=f"{self.icons['error']} Failed")
        self.completion_time_label.config(text="")
        self.process_details_label.config(text="Processing failed!")

        self.status_label.config(
            text=f"{self.icons['error']} e57_pointcloud_pipeline_v1.py failed",
            foreground="red"
        )

        self.reset_ui()

        messagebox.showerror(
            "❌ Pipeline Error",
            f"e57_pointcloud_pipeline_v1.py failed:\n\n{error_msg[:300]}{'...' if len(error_msg) > 300 else ''}"
        )

    def cancel_process(self):
        """Cancel the pipeline process"""
        if self.is_processing:
            self.is_processing = False
            if self.pipeline_process:
                try:
                    self.pipeline_process.terminate()
                    self.pipeline_process.wait(timeout=3)
                except:
                    try:
                        self.pipeline_process.kill()
                    except:
                        pass

            self.progress_percent_label.config(text="Cancelled", foreground="red")
            self.estimated_time_label.config(text=f"{self.icons['warning']} Stopped")
            self.completion_time_label.config(text="")
            self.process_details_label.config(text="Processing cancelled by user")

            self.status_label.config(
                text=f"{self.icons['warning']} e57_pointcloud_pipeline_v1.py process stopped",
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


# Main execution
if __name__ == "__main__":
    print("🚀 Starting E57 Point Cloud Pipeline Tool...")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        root = tk.Tk()
        print("✅ Root window created")

        app = FileImportApp(root)
        print("✅ App created, starting mainloop...")

        root.mainloop()
        print("✅ Application closed normally")

    except Exception as e:
        print(f"❌ Error running application: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")
