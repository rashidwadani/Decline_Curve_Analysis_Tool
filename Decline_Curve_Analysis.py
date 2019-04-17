import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import datetime

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure



class Window(Frame):
    
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master        
        self.master.title("Decline Curve Analysis | NeuDax")
        self.pack(fill=BOTH, expand=1, padx=20)

        self.summary_parameters = pd.DataFrame(columns=['API_NUMBER', 'qi', 'di', 'b', 'sqrt_error', 'skipped_days', 'TimeStamp'])       
        
        self.read_data()
        
        row = 0
        
        label_API = Label(self, text="Well API Number:")
        label_API.grid(row=row, column=0, sticky=E)
        self.API_value = StringVar()
        self.API = ttk.Combobox(self, textvariable=self.API_value, state='readonly')
        self.API['values'] = self.list_of_API_numbers
        self.API.current(0)
        self.API.grid(row=row, column=1, columnspan=2, sticky=W+E)        

        button_prev_well = ttk.Button(self, text="<<< Previous Well", command=lambda: self.go_to_prev_well())
        button_prev_well.grid(row=row, column=3, sticky=W+E)  
        button_next_well = ttk.Button(self, text="      Next Well >>>", command=lambda: self.go_to_next_well())
        button_next_well.grid(row=row, column=4, sticky=W+E)   
        
        label_typ_plot = Label(self, text="Type of Plot:")
        label_typ_plot.grid(row=row, column=5, sticky=E)
        self.typ_plot_value = StringVar()
        self.typ_plot = ttk.Combobox(self, textvariable=self.typ_plot_value, state='readonly')
        self.typ_plot['values'] = ['Linear Plot', 'Semi-Log Plot', 'Log-Log Plot', 'Rate vs. Cum']
        self.typ_plot.current(0)
        self.typ_plot.grid(row=row, column=6, sticky=W+E) 
        row += 1

        label_guide_uncertainty = Label(self, text="Uncertianty Guide Plots (%)")
        label_guide_uncertainty.grid(row=row, column=0, sticky=E)
        self.guide_uncertainty_percent = tk.Entry(self)
        self.guide_uncertainty_percent.grid(row=row, column=1, sticky=W+E)
        self.guide_uncertainty_percent.delete(0, END)
        self.guide_uncertainty_percent.insert(0, "0.2")
        self.guide_uncertainty_plot_var = BooleanVar()
        self.guide_uncertainty_plot = Checkbutton(self, text="Plot Guide Lines", variable=self.guide_uncertainty_plot_var, anchor=W, command=lambda: self.plot())
        self.guide_uncertainty_plot.grid(row=row, column=2, sticky=W+E) 
        row += 1
        
        label_skip_days=Label(self, text="Skip Data Until Day:")
        label_skip_days.grid(row=row, column=0, sticky=E)
        self.skip_days = tk.Entry(self)
        self.skip_days.grid(row=row, column=1, sticky=W+E)
        self.skip_days.delete(0, END)
        self.skip_days.insert(0, "0")
        row += 1


        button_load = ttk.Button(self, text="Step 1) Load Data and Reset Values", command=lambda: self.load_data())
        button_load.grid(row=row, column=1, columnspan=2, sticky=W+E)        
        row += 1
        
        label_lower_limit_t=Label(self, text="Time Interval (Days):")
        label_lower_limit_t.grid(row=row, column=0, sticky=E)
        self.lower_limit_t = tk.Entry(self)
        self.lower_limit_t.grid(row=row, column=1, sticky=W+E)
        self.upper_limit_t = tk.Entry(self)
        self.upper_limit_t.grid(row=row, column=2, sticky=W+E)
        row += 1

        label_lower_limit_qi=Label(self, text="Limit for qi:")
        label_lower_limit_qi.grid(row=row, column=0, sticky=E)
        self.lower_limit_qi = tk.Entry(self)
        self.lower_limit_qi.grid(row=row, column=1, sticky=W+E)
        self.upper_limit_qi = tk.Entry(self)
        self.upper_limit_qi.grid(row=row, column=2, sticky=W+E)

        row += 1

        label_lower_limit_di = Label(self, text="Limit for di:")
        label_lower_limit_di.grid(row=row, column=0, sticky=E)
        self.lower_limit_di = tk.Entry(self)
        self.lower_limit_di.grid(row=row, column=1, sticky=W+E)
        self.upper_limit_di = tk.Entry(self)
        self.upper_limit_di.grid(row=row, column=2, sticky=W+E)

        row += 1
        
        label_lower_limit_b = Label(self, text="Limit for b:")
        label_lower_limit_b.grid(row=row, column=0, sticky=E)
        self.lower_limit_b = tk.Entry(self)
        self.lower_limit_b.grid(row=row, column=1, sticky=W+E)
        self.upper_limit_b = tk.Entry(self)
        self.upper_limit_b.grid(row=row, column=2, sticky=W+E)

        row += 1        

        label_method = Label(self, text="Method:")
        label_method.grid(row=row, column=0, sticky=E)
        self.method_value = StringVar()
        self.method = ttk.Combobox(self, textvariable=self.method_value, state='readonly')
        self.method['values'] = ['trf', 'dogbox']
        self.method.current(0)
        self.method.grid(row=row, column=1, sticky=W+E)
        self.fit_on_cum_var = IntVar()
        self.fit_on_cum = Checkbutton(self, text="Fit on Cumulative", variable=self.fit_on_cum_var, anchor=W)
        self.fit_on_cum.grid(row=row, column=2, sticky=W+E)        
        row += 1
        

        label_loss = Label(self, text="Loss Function:")
        label_loss.grid(row=row, column=0, sticky=E)
        self.loss_value = StringVar()
        self.loss = ttk.Combobox(self, textvariable=self.loss_value, state='readonly')
        self.loss['values'] = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']
        self.loss.current(0)
        self.loss.grid(row=row, column=1, sticky=W+E)
        self.auto_fit_var = BooleanVar()
        self.auto_fit = Checkbutton(self, text="Auto Fit", variable=self.auto_fit_var, anchor=W)
        self.auto_fit.grid(row=row, column=2, sticky=W+E)  
        row += 1


        button_fit_hyp = ttk.Button(self, text="Step 2) Fit Hyperbolic", command=lambda: self.fit())
        button_fit_hyp.grid(row=row, column=1, columnspan=2, sticky=W+E)
        row += 1

        
        label_qi = Label(self, text="Initial Rate (qi):")
        label_qi.grid(row=row, column=0, sticky=E)
        self.par_qi = tk.Entry(self)
        self.par_qi.grid(row=row, column=1, sticky=W+E)

        
        label_qi = Label(self, text="Previous Parameters:")
        label_qi.grid(row=row, column=2, sticky=W+E)
        row += 1
        
        
        label_di = Label(self, text="Initial Decline Rate (di):")
        label_di.grid(row=row, column=0, sticky=E)
        self.par_di = tk.Entry(self)
        self.par_di.grid(row=row, column=1, sticky=W+E)

        
        self.prev_par_value = StringVar()
        self.prev_par = ttk.Combobox(self, textvariable=self.prev_par_value, state='readonly')
        self.prev_par.grid(row=row, column=2, sticky=W+E)
        row += 1
        
        label_b = Label(self, text="Curvature (b):")
        label_b.grid(row=row, column=0, sticky=E)
        self.par_b = tk.Entry(self)
        self.par_b.grid(row=row, column=1, sticky=W+E)

        
        button_prev_par = ttk.Button(self, text="Load Previous Parameters", command=lambda: self.load_prev_par())
        button_prev_par.grid(row=row, column=2, sticky=W+E)
        row += 1

        self.label_error1 = Label(self, text='Root Mean Squared Error:')
        self.label_error1.grid(row=row, column=1, sticky=E)

        
        row += 3
                     
        button_plot = ttk.Button(self, text="Step 3) Manual Modification", command=lambda: self.plot())
        button_plot.grid(row=row, column=1, columnspan=2, sticky=W+E)
        row += 1
        
        
        button_save = ttk.Button(self, text="Step 4) Save", command=lambda: self.save())
        button_save.grid(row=row, column=1, columnspan=2, sticky=W+E)
        row += 1
        
        button_exit = ttk.Button(self, text="Exit", command=lambda: self.quit())
        button_exit.grid(row=row, column=1, columnspan=2, sticky=W+E)
        row += 1

        self.fr = tk.Frame(self)
        self.fr.grid(row=1, column=3, rowspan=26, columnspan=4, sticky=N+E+W+S)
        row += 1
        
        self.f, self.ax = plt.subplots()        
        self.canvas = FigureCanvasTkAgg(self.f, master=self.fr)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.fr)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('button_press_event', self.selecting_point)        
        
        self.load_data()
        
        self.API.bind("<<ComboboxSelected>>", lambda _ : self.API_changes())
        self.prev_par.bind("<<ComboboxSelected>>", lambda _ : self.load_prev_par())
        self.typ_plot.bind("<<ComboboxSelected>>", lambda _ : self.plot())    
    
    def read_data(self):
        #input_file_path =  filedialog.askopenfilename(title = "Select file to open",filetypes = (("csv files","*.csv"),("all files","*.*")))
        input_file_path = 'Small_Dataset_of_Good_APIs.csv'
        self.production_data = pd.read_csv(input_file_path)    
        temp_list_of_APIs = self.production_data['API_NUMBER'].drop_duplicates()
        self.list_of_API_numbers = temp_list_of_APIs.tolist()  
        
    def load_data(self):
        
        try:
            self.skipped_days = float(self.skip_days.get())
        except:
            self.skipped_days = 0
        
        selected_production_data = pd.DataFrame()
        selected_production_data = self.production_data.loc[self.production_data['API_NUMBER']==float(self.API.get()), 
                                                              ['OIL_PROD_DAYS', 'OIL_PRODUCTION_BBL']]
        selected_production_data = selected_production_data.dropna()
        selected_production_data = selected_production_data.loc[selected_production_data['OIL_PRODUCTION_BBL']>0]
        selected_production_data['RATE_PROD'] = selected_production_data['OIL_PRODUCTION_BBL']/selected_production_data['OIL_PROD_DAYS']
        selected_production_data['CUM_DAYS'] = selected_production_data['OIL_PROD_DAYS'].cumsum()
        selected_production_data['CUM_PROD'] = selected_production_data.loc[selected_production_data['CUM_DAYS']>self.skipped_days, 'OIL_PRODUCTION_BBL'].cumsum()
        
        self.t_cum = selected_production_data.loc[selected_production_data['CUM_DAYS']>self.skipped_days, 'CUM_DAYS'] - self.skipped_days
        self.t = self.middle_t(self.t_cum)
        self.q = selected_production_data.loc[selected_production_data['CUM_DAYS']>self.skipped_days, 'RATE_PROD']
        
        dt = np.copy(self.t_cum)
        dt[1:] = np.array(self.t)[1:]-np.array(self.t)[:-1]         
        self.q_cum = np.cumsum(dt*np.array(self.q))
        
        print('t', self.t)
        print('q', self.q)
        print('t_cum', self.t_cum)
        print('q_cum', self.q_cum)
        
        self.selected_t = np.array([])
        self.selected_q = np.array([])
        self.selected_t_cum = np.array([])
        self.selected_q_cum = np.array([])
        self.valid_t = np.array(self.t)
        self.valid_q = np.array(self.q)
        self.valid_t_cum = np.array(self.t_cum)
        self.valid_q_cum = np.array(self.q_cum)
        
        self.par_qi.delete(0, END)
        self.par_qi.insert(0, "0.0")
        self.par_di.delete(0, END)
        self.par_di.insert(0, "1.0")
        self.par_b.delete(0, END)
        self.par_b.insert(0, "0.5")
        self.lower_limit_t.delete(0, END)
        self.lower_limit_t.insert(0, "0")
        self.upper_limit_t.delete(0, END)
        self.upper_limit_t.insert(0, str(self.t.max()))
        self.lower_limit_qi.delete(0, END)
        self.lower_limit_qi.insert(0, "0.0")
        self.upper_limit_qi.delete(0, END)
        self.upper_limit_qi.insert(0, "50000.0")
        self.lower_limit_di.delete(0, END)
        self.lower_limit_di.insert(0, "0.0")
        self.upper_limit_di.delete(0, END)
        self.upper_limit_di.insert(0, "1.0")
        self.lower_limit_b.delete(0, END)
        self.lower_limit_b.insert(0, "0.001")
        self.upper_limit_b.delete(0, END)
        self.upper_limit_b.insert(0, "2.0")
        self.prev_par.set('')
                
        if self.auto_fit_var.get():
            self.fit()
        else:
            self.plot()
        
        self.load_prev_timestamps()
        
        
    def load_prev_timestamps(self):
        # Loading previous analysis and parameters (if exists)
        previous_parameters = pd.DataFrame()
        previous_parameters = self.summary_parameters.loc[self.summary_parameters['API_NUMBER']==self.API.get(), :]
        temp_list_of_prev_par = []
        temp_list_of_prev_par = previous_parameters['TimeStamp'].drop_duplicates()
        self.list_of_prev_par = temp_list_of_prev_par.tolist()
        self.prev_par['values'] = self.list_of_prev_par
        
    
    def load_prev_par(self):
        previous_parameters = pd.DataFrame()
        previous_parameters = self.summary_parameters.loc[(self.summary_parameters['API_NUMBER']==self.API.get()) & (self.summary_parameters['TimeStamp']==self.prev_par.get()), :]
        
        self.skip_days.delete(0, END)
        self.skip_days.insert(0, previous_parameters['skipped_days'].values[0])
        
        self.load_data()
        
        self.par_qi.delete(0, END)
        self.par_qi.insert(0, previous_parameters['qi'].values[0])
        self.par_di.delete(0, END)
        self.par_di.insert(0, previous_parameters['di'].values[0])
        self.par_b.delete(0, END)
        self.par_b.insert(0, previous_parameters['b'].values[0])
        
        self.plot()
        

    def save(self):
        
        self.summary_parameters = self.summary_parameters.append({'API_NUMBER':self.API.get(), 'qi': self.par_qi.get(),  
                                                                  'di':self.par_di.get(), 'b': self.par_b.get(),
                                                                  'sqrt_error': self.sqrt_error,
                                                                  'skipped_days': self.skipped_days,
                                                                  'TimeStamp': self.now_code2()}, ignore_index=True)
        now_code = self.now_code()
        par_filepath = './save/saved_parameters_'+str(now_code)+'.csv'
        self.summary_parameters.to_csv(par_filepath, index=False)
        
        self.load_prev_timestamps()

    
    def now_code(self):
        now = datetime.datetime.now()
        now_code = now.year*10000000000+now.month*100000000+now.day*1000000+now.hour*10000+now.minute*100+now.second
        return now_code
 
    
    def now_code2(self):
        now = datetime.datetime.now()
        now_code = str(now.year)+'/'+str(now.month)+'/'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        return now_code

    
    def plot(self):
        self.popt = [float(self.par_qi.get()), float(self.par_di.get()), float(self.par_b.get())]
        self.t_model = np.arange(int(np.max(self.t)))
        # This q_model will be plotted as model interpolation
        self.q_model = self.func_hyp(self.t_model, *self.popt)
        
        # We increase and decrease "b" factor by 10% as guiding plots
        self.q_model_qi_low = self.func_hyp(self.t_model, self.popt[0]*(1-float(self.guide_uncertainty_percent.get())), self.popt[1], self.popt[2])
        self.q_model_qi_up = self.func_hyp(self.t_model, self.popt[0]*(1+float(self.guide_uncertainty_percent.get())), self.popt[1], self.popt[2])
        self.q_model_di_low = self.func_hyp(self.t_model, self.popt[0], self.popt[1]*(1-float(self.guide_uncertainty_percent.get())), self.popt[2])
        self.q_model_di_up = self.func_hyp(self.t_model, self.popt[0], self.popt[1]*(1+float(self.guide_uncertainty_percent.get())), self.popt[2])       
        self.q_model_b_low = self.func_hyp(self.t_model, self.popt[0], self.popt[1], self.popt[2]*(1-float(self.guide_uncertainty_percent.get())))
        self.q_model_b_up = self.func_hyp(self.t_model, self.popt[0], self.popt[1], self.popt[2]*(1+float(self.guide_uncertainty_percent.get())))
        
        # This q_model2 will be used for error calculation
        self.q_model2 = self.func_hyp(self.valid_t, *self.popt)
        
        
        # Here, we compute cum model
        self.q_cum_model = self.func_cum_hyp(self.t_model, *self.popt)
        
        # We increase and decrease "b" factor by 10% as guiding plots (for cumulative plots)
        self.q_cum_model_qi_low = self.func_cum_hyp(self.t_model, self.popt[0]*(1-float(self.guide_uncertainty_percent.get())), self.popt[1], self.popt[2])
        self.q_cum_model_qi_up = self.func_cum_hyp(self.t_model, self.popt[0]*(1+float(self.guide_uncertainty_percent.get())), self.popt[1], self.popt[2])
        self.q_cum_model_di_low = self.func_cum_hyp(self.t_model, self.popt[0], self.popt[1]*(1-float(self.guide_uncertainty_percent.get())), self.popt[2])
        self.q_cum_model_di_up = self.func_cum_hyp(self.t_model, self.popt[0], self.popt[1]*(1+float(self.guide_uncertainty_percent.get())), self.popt[2])       
        self.q_cum_model_b_low = self.func_cum_hyp(self.t_model, self.popt[0], self.popt[1], self.popt[2]*(1-float(self.guide_uncertainty_percent.get())))
        self.q_cum_model_b_up = self.func_cum_hyp(self.t_model, self.popt[0], self.popt[1], self.popt[2]*(1+float(self.guide_uncertainty_percent.get())))

        t_lim = np.array([float(self.lower_limit_t.get()), float(self.lower_limit_t.get()), 0.0])
        q_lim = np.array([0.0, float(self.par_qi.get()), float(self.par_qi.get())])
        
        self.f.clear()
        
        if self.typ_plot.get() == "Rate vs. Cum":
            self.ax = plt.semilogy(self.q_cum, self.q, 'g.')
            self.ax = plt.semilogy(self.q_cum_model, self.q_model, 'k-')
            self.ax = plt.semilogy(self.selected_q_cum, self.selected_q, 'kx')
            plt.gca().set_xlabel('Cumulative Rate (Bbl)')
            plt.gca().set_ylabel('Rate (Bbl)')
            
        else:
            if self.typ_plot.get() == "Linear Plot":
                self.ax = plt.plot(self.t, self.q, 'b.', label='Rate Data') 
                self.ax = plt.plot(self.selected_t, self.selected_q, 'kx')
                self.ax = plt.plot(t_lim, q_lim, 'g-') 
                self.ax = plt.plot(self.t_model, self.q_model, 'r-', linewidth=2, label='Rate Model')
                if self.guide_uncertainty_plot_var.get():
                    self.ax = plt.plot(self.t_model, self.q_model_b_low, 'r:', linewidth=1)
                    self.ax = plt.plot(self.t_model, self.q_model_b_up, 'r-.', linewidth=1)
                    self.ax = plt.plot(self.t_model, self.q_model_qi_low, 'b:', linewidth=1)
                    self.ax = plt.plot(self.t_model, self.q_model_qi_up, 'b-.', linewidth=1)
                    self.ax = plt.plot(self.t_model, self.q_model_di_low, 'g:', linewidth=1)
                    self.ax = plt.plot(self.t_model, self.q_model_di_up, 'g-.', linewidth=1)
                
            elif self.typ_plot.get() == "Semi-Log Plot":
                self.ax = plt.semilogy(self.t, self.q, 'b.') 
                self.ax = plt.semilogy(self.selected_t, self.selected_q, 'kx')
                self.ax = plt.semilogy(t_lim, q_lim, 'g-')
                self.ax = plt.semilogy(self.t_model, self.q_model, 'r-', linewidth=2)
                if self.guide_uncertainty_plot_var.get():
                    self.ax = plt.semilogy(self.t_model, self.q_model_b_low, 'r:', linewidth=1)
                    self.ax = plt.semilogy(self.t_model, self.q_model_b_up, 'r-.', linewidth=1)
                    self.ax = plt.semilogy(self.t_model, self.q_model_qi_low, 'b:', linewidth=1)
                    self.ax = plt.semilogy(self.t_model, self.q_model_qi_up, 'b-.', linewidth=1)
                    self.ax = plt.semilogy(self.t_model, self.q_model_di_low, 'g:', linewidth=1)
                    self.ax = plt.semilogy(self.t_model, self.q_model_di_up, 'g-.', linewidth=1)
                    
            elif self.typ_plot.get() == "Log-Log Plot":
                self.ax = plt.loglog(self.t, self.q, 'b.')
                self.ax = plt.semilogy(self.selected_t, self.selected_q, 'kx')
                self.ax = plt.loglog(t_lim, q_lim, 'g-') 
                self.ax = plt.loglog(self.t_model, self.q_model, 'r-', linewidth=2)
                if self.guide_uncertainty_plot_var.get():
                    self.ax = plt.loglog(self.t_model, self.q_model_b_low, 'r:', linewidth=1)
                    self.ax = plt.loglog(self.t_model, self.q_model_b_up, 'r-.', linewidth=1)
                    self.ax = plt.loglog(self.t_model, self.q_model_qi_low, 'b:', linewidth=1)
                    self.ax = plt.loglog(self.t_model, self.q_model_qi_up, 'b-.', linewidth=1)
                    self.ax = plt.loglog(self.t_model, self.q_model_di_low, 'g:', linewidth=1)
                    self.ax = plt.loglog(self.t_model, self.q_model_di_up, 'g-.', linewidth=1)
                
            
            plt.gca().set_xlabel('Time (Days)')
            plt.gca().set_ylabel('Rate (Bbl)')
            plt.gca().set_title(self.API.get()+" | qi:"+self.par_qi.get()+" | di:"+self.par_di.get()+" | b:"+self.par_b.get()+" | "+self.now_code2(), fontsize=8)
            
            self.ax2 = plt.gca().twinx()
            self.ax2 = plt.plot(self.t_cum, self.q_cum, 'c+', label='Cum Data') 
            self.ax2 = plt.plot(self.t_model, self.q_cum_model, 'm-', linewidth=2, label='Cum Model')
            
            if self.guide_uncertainty_plot_var.get():
                self.ax2 = plt.plot(self.t_model, self.q_cum_model_b_low, 'r:', linewidth=1)
                self.ax2 = plt.plot(self.t_model, self.q_cum_model_b_up, 'r-.', linewidth=1)
                self.ax2 = plt.plot(self.t_model, self.q_cum_model_qi_low, 'b:', linewidth=1)
                self.ax2 = plt.plot(self.t_model, self.q_cum_model_qi_up, 'b-.', linewidth=1)
                self.ax2 = plt.plot(self.t_model, self.q_cum_model_di_low, 'g:', linewidth=1)
                self.ax2 = plt.plot(self.t_model, self.q_cum_model_di_up, 'g-.', linewidth=1)
                
        self.f.tight_layout()
        self.canvas.draw()
        
        self.sqrt_error = self.error()
        self.label_error2 = Label(self, text='                        ')
        self.label_error2.grid(row=14, column=2, sticky=W)
        self.label_error2 = Label(self, text=str(round(self.sqrt_error,5)))
        self.label_error2.grid(row=14, column=2, sticky=W)
        
        self.forecast()
    
    def error(self):
        diff = (self.q_model2-self.valid_q)**2
        return (diff.mean())**0.5

    def middle_t(self, t):
        b = np.append(np.array([0]), np.array(t))
        return 0.5*(b[:-1]+b[1:])
    
    def func_hyp(self, t, qi, di, b):
            return qi/((1+di*b*t)**(1/b))
    
    # From Fekete
    def func_cum_hyp(self, t, qi, di, b):
        return (qi/((1-b)*di))*(1-(1+b*di*t)**(1-(1/b)))
    
    def func_duong(self, t, qi, a, m):
        return qi*t**(-m)*np.exp((a)/(1-m)*(t**(1-m))-1)
    
    def fit(self):
        
    
        t = self.valid_t[(self.valid_t<float(self.upper_limit_t.get()))]
        q = self.valid_q[(self.valid_t<float(self.upper_limit_t.get()))]
        
        t_cum = self.valid_t_cum[(self.valid_t<float(self.upper_limit_t.get()))]
        q_cum = self.valid_q_cum[(self.valid_t<float(self.upper_limit_t.get()))]
        print(t_cum)

        lo_qi = float(self.lower_limit_qi.get())
        up_qi = float(self.upper_limit_qi.get())
        lo_di = float(self.lower_limit_di.get())
        up_di = float(self.upper_limit_di.get())
        lo_b  = float(self.lower_limit_b.get())
        up_b  = float(self.upper_limit_b.get())
        method = self.method.get()
        loss = self.loss.get()
        
        if self.fit_on_cum_var.get()==0:
            self.popt, self.pcov = curve_fit(f=self.func_hyp, xdata=t, ydata=q, check_finite=True, 
                                             method=method, loss=loss,
                                             bounds=([lo_qi, lo_di, lo_b], [up_qi, up_di, up_b]))
        
        elif self.fit_on_cum_var.get()==1:  
            self.popt, self.pcov = curve_fit(f=self.func_cum_hyp, xdata=t_cum, ydata=q_cum, check_finite=True, 
                                             method=method, loss=loss,
                                             bounds=([lo_qi, lo_di, lo_b], [up_qi, up_di, up_b]))
      
        self.popt = np.round(self.popt, decimals=5)
        
        self.q_model = self.func_hyp(self.t, *self.popt)
        self.par_qi.delete(0, END)
        self.par_qi.insert(0, self.popt[0])
        self.par_di.delete(0, END)
        self.par_di.insert(0, self.popt[1])
        self.par_b.delete(0, END)
        self.par_b.insert(0, self.popt[2])
        self.plot()
        
    
    def forecast(self):
        days = np.array([90.0, 180.0, 365.0])
        qi = float(self.par_qi.get())
        di = float(self.par_di.get())
        b = float(self.par_b.get())
        self.production_forecast = np.round(self.func_cum_hyp(days, qi, di, b), decimals=0)
        EUR = np.round(self.EUR(), decimals=0)
        production_statement = "IP90:"+str(self.production_forecast[0])+" | IP180:"+str(self.production_forecast[1])+" | IP365:"+str(self.production_forecast[2])+" | EUR:"+ str(EUR)
        
        self.label_forecast = Label(self, text="                                                    ")
        self.label_forecast.grid(row=15, column=1, columnspan=2, sticky=E+W) 
        self.label_forecast = Label(self, text=production_statement)
        self.label_forecast.grid(row=15, column=1, columnspan=2, sticky=E+W)  
    
    def go_to_next_well(self):
        try:
            current_well_index = self.API.current()
            self.API.current(current_well_index+1)
            self.reset_skip_days()
            self.load_data()
        except:
            return
    
    def go_to_prev_well(self):
        try:
            current_well_index = self.API.current()
            self.API.current(current_well_index-1)
            self.reset_skip_days()
            self.load_data()
        except:
            return

    def API_changes(self):
        self.reset_skip_days()
        self.load_data()

   
    def reset_skip_days(self):
        self.skip_days.delete(0, END)
        self.skip_days.insert(0, "0.0")
            
    
    def selecting_point(self, event):
        
        if (event.button==1) and (event.dblclick==True):
            self.skip_days.delete(0, END)
            self.skip_days.insert(0, "0")
            self.load_data()
        
        if self.typ_plot.get() == "Rate vs. Cum":
            return
        else:
            if (event.button==3) and (event.dblclick==False):
                current_skip_days = float(self.skip_days.get())
                self.skip_days.delete(0, END)
                self.skip_days.insert(0, str(current_skip_days+int(event.xdata)))
                self.load_data()
            elif (event.button==1) and (event.dblclick==False):
                try:
                    self.find_nearest(event.xdata)
                    self.plot()
                except:
                    return
    
    def find_nearest(self, value):
        vector_t = np.copy(self.valid_t)
        vector_q = np.copy(self.valid_q)
        vector_t_cum = np.copy(self.valid_t_cum)
        vector_q_cum = np.copy(self.valid_q_cum)
        idx = (np.abs(vector_t - value)).argmin()
        self.selected_t = np.append(self.selected_t,vector_t[idx])
        self.selected_q = np.append(self.selected_q,vector_q[idx])
        self.selected_t_cum = np.append(self.selected_t_cum,vector_t_cum[idx])
        self.selected_q_cum = np.append(self.selected_q_cum,vector_q_cum[idx])
        self.valid_t = np.delete(self.valid_t, idx)
        self.valid_q = np.delete(self.valid_q, idx)
        self.valid_t_cum = np.delete(self.valid_t_cum, idx)
        self.valid_q_cum = np.delete(self.valid_q_cum, idx)

                
    def EUR(self):
        qi = float(self.par_qi.get())
        di = float(self.par_di.get())
        b = float(self.par_b.get())
        q_lowest_for_EUR = 1.0
        for t in range(1000000):
            if self.func_hyp(t, qi, di, b)<=q_lowest_for_EUR:
                return self.func_cum_hyp(t, qi, di, b)
            

                
    
    def quit(self):
        global root
        root.destroy()
        root.quit()

        
root = Tk()
#root.geometry("620x790")
app = Window(root)
root.lift()
root.attributes('-topmost', True)
root.attributes('-topmost', False)
root.mainloop()