import tkinter as tk
import matplotlib.pyplot as plt

from algo import LightsDesignerAlgorithm

RECOMMENDED_LUX_TABLE = {
    "Sala": 150,
    "Dormitorio": 100,
    "Cocina": 300,
    "Baño": 300,
}

LIGHT_TYPE = {
    "Clara": "White",
    "Calida": "Warm",

}


class DesignerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Designer UI")
        self.root.geometry("800x600")

        general_frame = tk.Frame(self.root)
        general_frame.pack()

        width_label = tk.Label(general_frame, text="Largo")
        width_label.grid(row=0, column=0)
        self.width_entry = tk.Entry(general_frame)
        self.width_entry.grid(row=0, column=1)

        height_label = tk.Label(general_frame, text="Ancho")
        height_label.grid(row=1, column=0)
        self.height_entry = tk.Entry(general_frame)
        self.height_entry.grid(row=1, column=1)

        grid_width_label = tk.Label(general_frame, text="Grid Largo")
        grid_width_label.grid(row=2, column=0)
        self.grid_width_entry = tk.Entry(general_frame)
        self.grid_width_entry.grid(row=2, column=1)

        grid_height_label = tk.Label(general_frame, text="Grid Ancho")
        grid_height_label.grid(row=3, column=0)
        self.grid_height_entry = tk.Entry(general_frame)
        self.grid_height_entry.grid(row=3, column=1)

        light_type_label = tk.Label(general_frame, text="Tipo de luz")
        light_type_label.grid(row=4, column=0)
        # types: Clara, Calida
        self.light_type = tk.StringVar()
        self.light_type.set("Clara")
        light_type_option = tk.OptionMenu(
            general_frame, self.light_type, "Clara", "Calida"
        )
        light_type_option.grid(row=4, column=1)

        area_type_label = tk.Label(general_frame, text="Tipo de área")
        area_type_label.grid(row=5, column=0)
        self.area_type = tk.StringVar()
        self.area_type.set("Sala")
        area_type_option = tk.OptionMenu(
            general_frame, self.area_type, "Sala", "Dormitorio", "Cocina", "Baño"
        )
        area_type_option.grid(row=5, column=1)

        init_population_label = tk.Label(
            general_frame, text="Numero de poblacion inicial"
        )
        init_population_label.grid(row=6, column=0)
        self.init_population_entry = tk.Entry(general_frame)
        self.init_population_entry.grid(row=6, column=1)

        max_population_label = tk.Label(general_frame, text="Maximo de poblacion")
        max_population_label.grid(row=7, column=0)
        self.max_population_entry = tk.Entry(general_frame)
        self.max_population_entry.grid(row=7, column=1)

        prob_mutation_label = tk.Label(general_frame, text="Probabilidad de mutacion")
        prob_mutation_label.grid(row=8, column=0)
        self.prob_mutation_entry = tk.Entry(general_frame)
        self.prob_mutation_entry.grid(row=8, column=1)

        prob_mutation_per_grid_label = tk.Label(
            general_frame, text="Probabilidad de mutacion por grid"
        )
        prob_mutation_per_grid_label.grid(row=9, column=0)
        self.prob_mutation_per_grid_entry = tk.Entry(general_frame)
        self.prob_mutation_per_grid_entry.grid(row=9, column=1)

        prob_crossover_label = tk.Label(general_frame, text="Probabilidad de crossover")
        prob_crossover_label.grid(row=10, column=0)
        self.prob_crossover_entry = tk.Entry(general_frame)
        self.prob_crossover_entry.grid(row=10, column=1)

        generations_label = tk.Label(general_frame, text="Generaciones")
        generations_label.grid(row=11, column=0)
        self.generations_entry = tk.Entry(general_frame)
        self.generations_entry.grid(row=11, column=1)

        # show the following buttons
        # Empezar
        # Salir

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        start_button = tk.Button(button_frame, text="Empezar", command=self.start)
        start_button.pack(side=tk.RIGHT)

        exit_button = tk.Button(button_frame, text="Salir", command=self.exit)
        exit_button.pack(side=tk.LEFT)

        self.root.bind("q", lambda event: self.exit())

    def start(self):
        # get values from the inputs
        width = int(self.width_entry.get())
        height = int(self.height_entry.get())
        grid_width = int(self.grid_width_entry.get())
        grid_height = int(self.grid_height_entry.get())
        light_type = self.light_type.get()
        area_type = self.area_type.get()
        init_population = int(self.init_population_entry.get())
        max_population = int(self.max_population_entry.get())
        prob_mutation = float(self.prob_mutation_entry.get())
        prob_mutation_per_grid = float(self.prob_mutation_per_grid_entry.get())
        prob_crossover = float(self.prob_crossover_entry.get())
        generations = int(self.generations_entry.get())

        algo = LightsDesignerAlgorithm(
            width,
            height,
            grid_width,
            grid_height,
            LIGHT_TYPE[light_type],
            RECOMMENDED_LUX_TABLE[area_type],
            init_population,
            max_population,
            prob_mutation,
            prob_mutation_per_grid,
            prob_crossover,
            generations,
        )
        population_history, statistics_history = algo.run()

        # graficar historial de estadisticas
        stats_figure, stats_ax = plt.subplots()
        stats_figure.canvas.manager.set_window_title("Historial de estadisticas")
        stats_ax.set_title("Historial de estadisticas")
        stats_ax.set_xlabel("Generaciones")
        stats_ax.set_ylabel("Fitness")
        stats_ax.plot([x["best"] for x in statistics_history], label="Mejor")
        stats_ax.plot([x["worst"] for x in statistics_history], label="Peor")
        stats_ax.plot([x["mean"] for x in statistics_history], label="Promedio")
        stats_ax.legend()

        # graficar el mejor individuo de la ultima generacion
        # este representa el mejor diseño de iluminacion, por lo que se debe primero graficar el grid
        best_system = population_history[-1][0]
        best_system_grids = best_system["grids"]
        best_system_mean = best_system["mean"]
        best_system_standard_deviation = best_system["standard_deviation"]
        best_system_fitness = best_system["fitness"]

        best_system_figure, best_system_ax = plt.subplots()
        best_system_figure.canvas.manager.set_window_title(
            "Mejor sistema de iluminacion"
        )
        best_system_ax.set_title("Mejor sistema de iluminacion")
        best_system_ax.set_xlabel("Ancho")
        best_system_ax.set_ylabel("Alto")
        best_system_ax.set_xlim(0, width)
        best_system_ax.set_ylim(0, height)
        # draw grid
        for i in range(grid_width):
            best_system_ax.axvline(i * (width / grid_width), color="black")
        for i in range(grid_height):
            best_system_ax.axhline(i * (height / grid_height), color="black")

        # draw bulbs
        for grid in best_system_grids:
            best_system_ax.scatter(
                grid["x"], grid["y"], s=grid["bulb"].lumens, color="yellow"
            )


        # graficar los luxes de cada grid del mejor sistema
        luxes_figure, luxes_ax = plt.subplots()
        luxes_figure.canvas.manager.set_window_title("Luxes por grid del mejor sistema")
        luxes_ax.set_title("Luxes por grid del mejor sistema")
        luxes_ax.set_xlabel("Grid")
        luxes_ax.set_ylabel("Luxes")
        luxes_ax.plot([x["luxes"] for x in best_system_grids])

        # crear ventana y mostrar configuracion del sistema
        config_window = tk.Toplevel(self.root)
        config_window.title("Configuracion del sistema")
        config_window.geometry("800x500")

        config_text = tk.Text(config_window, font=("Arial", 18))
        config_text.pack()
        # mostrar focos por grid
        total_wattage = sum([grid["bulb"].wattage for grid in best_system_grids])
        total_lumens = sum([grid["bulb"].lumens for grid in best_system_grids])

        config_text.insert(
            tk.END,
            f"Generaciones: {generations}\n"
            f"Fitness: {best_system_fitness}\n"
            f"Promedio: {best_system_mean}\n"
            f"Desviacion estandar: {best_system_standard_deviation}\n"
            f"Total wattage: {total_wattage}\n"
            f"Total lumens: {total_lumens}\n"
        )
        for i, grid in enumerate(best_system_grids):
            config_text.insert(
                tk.END,
                f"Grid {i + 1}\n"
                f"X: {grid['x']}\n"
                f"Y: {grid['y']}\n"
                f"Tipo de foco: {grid['bulb'].light_type}\n"
                f"Wattage: {grid['bulb'].wattage}\n"
                f"Lumens: {grid['bulb'].lumens}\n"
                f"Units per package: {grid['bulb'].units_per_package}\n"
                f"Price per package: {grid['bulb'].price_per_package}\n"
                f"URL: {grid['bulb'].url}\n"
                f"Luxes: {grid['luxes']:.2f}\n\n",
            )

        config_text.config(state='disabled')



        plt.show()

        config_window.mainloop()

    def exit(self):
        print("Exit")
        self.root.destroy()
