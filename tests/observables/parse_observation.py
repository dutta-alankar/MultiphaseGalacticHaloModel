# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:12:15 2023

@author: alankar
"""
import numpy as np
from typing import Optional


class observedColDens:
    def __init__(
        self: "observedColDens",
        galaxyDataFile: str = "apjs456058t3_mrt.txt",
        galaxySizeFile: str = "VirialRad.txt",
    ) -> None:
        self.galaxyDataFile = galaxyDataFile
        self.galaxySizeFile = galaxySizeFile

    def _read_data(self: "observedColDens", filename: str) -> None:
        lines = None
        start = 41
        with open(filename, "r") as file:
            lines = file.readlines()

        galaxies = []
        names = []
        impact = []
        limit = []
        coldens = []
        e_coldens = []
        for line in lines[start:]:
            # print(line)
            galaxies.append(line[:17].strip())
            names.append(line[29:35])
            impact.append(float(line[25:28]))
            if line[78] == " ":
                limit.append("e")
            elif line[78] == "<":
                limit.append("l")
            elif line[78] == ">":
                limit.append("g")
            if line[80:85] != "     ":
                coldens.append(float(line[80:85]))
                if limit[-1] == "e":
                    e_coldens.append(float(line[86:90]))
                else:
                    e_coldens.append(0.0)
            else:
                if line[62] == " ":
                    limit[-1] = "e"
                elif line[62] == "<":
                    limit[-1] = "l"
                elif line[62] == ">":
                    limit[-1] = "g"
                coldens.append(float(line[64:69]))
                if limit[-1] == "e":
                    e_coldens.append(float(line[70:74]))
                else:
                    e_coldens.append(0.0)

        self.galaxies = galaxies
        self.names = names
        self.impact = impact
        self.limit = limit
        self.coldens = coldens
        self.e_coldens = e_coldens

    def _read_rvir(self: "observedColDens", filename: str) -> None:
        lines = None
        start = 6
        stop = 50
        with open(filename, "r") as file:
            lines = file.readlines()

        qsoGal_names = []
        rvir = []
        for line in lines[start:stop]:
            # print(line)
            values = line.split()
            qsoGal_names.append((values[0] + "_" + values[1]).strip())
            rvir.append(float(values[-1]))

        self.qsoGal_names = qsoGal_names
        self.rvir = rvir

    def col_density_gen(
        self: "observedColDens", element: Optional[str] = "O VI"
    ) -> tuple:
        self._read_data(self.galaxyDataFile)
        self._read_rvir(self.galaxySizeFile)

        indices = [
            index
            for index in range(len(self.names))
            if self.names[index].strip() == element
        ]
        _coldens_min = []
        _coldens_max = []
        _coldens_detect = []
        _e_coldens_detect = []
        gal_id_min = []
        gal_id_max = []
        gal_id_detect = []
        _impact_select_min = []
        _rvir_select_min = []
        _impact_select_max = []
        _rvir_select_max = []
        _impact_select_detect = []
        _rvir_select_detect = []

        for indx in indices:
            if self.limit[indx] == "l":
                gal_id_max.append(self.galaxies[indx])
                _coldens_max.append(self.coldens[indx])
                _impact_select_max.append(self.impact[indx])
                _rvir_select_max.append(
                    self.rvir[self.qsoGal_names.index(self.galaxies[indx])]
                )
            elif self.limit[indx] == "g":
                gal_id_min.append(self.galaxies[indx])
                _coldens_min.append(self.coldens[indx])
                _impact_select_min.append(self.impact[indx])
                _rvir_select_min.append(
                    self.rvir[self.qsoGal_names.index(self.galaxies[indx])]
                )
            elif self.limit[indx] == "e":
                gal_id_detect.append(self.galaxies[indx])
                _coldens_detect.append(self.coldens[indx])
                _e_coldens_detect.append(self.e_coldens[indx])
                _impact_select_detect.append(self.impact[indx])
                _rvir_select_detect.append(
                    self.rvir[self.qsoGal_names.index(self.galaxies[indx])]
                )

        impact_select_min = np.array(_impact_select_min)
        rvir_select_min = np.array(_rvir_select_min)
        impact_select_max = np.array(_impact_select_max)
        rvir_select_max = np.array(_rvir_select_max)
        impact_select_detect = np.array(_impact_select_detect)
        rvir_select_detect = np.array(_rvir_select_detect)
        coldens_min = np.array(_coldens_min)
        coldens_max = np.array(_coldens_max)
        coldens_detect = np.array(_coldens_detect)
        e_coldens_detect = np.array(_e_coldens_detect)

        return (
            gal_id_min,
            gal_id_max,
            gal_id_detect,
            rvir_select_min,
            rvir_select_max,
            rvir_select_detect,
            impact_select_min,
            impact_select_max,
            impact_select_detect,
            coldens_min,
            coldens_max,
            coldens_detect,
            e_coldens_detect,
        )
