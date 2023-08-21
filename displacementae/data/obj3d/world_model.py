import os
import platform
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL']='egl' 
# os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
from typing import List, Optional, Tuple, Union
import mujoco
import numpy as np
import numpy.typing as npt

class WorldModel:
    """
    This class maintains a world model of a colored object as it changes orientation and color.
    """
    def __init__(self, object_dir:str, object_name:str, figsize:Tuple[int,int]=(72,72)):
        """
        Initialize the mujoco simulation and renderer.

        Todo: add color option, 1D rotation option.
        """
        object_dir = os.path.expanduser(object_dir)
        xml = self._create_xml(object_dir, object_name)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        
        self._camera = self._model.body('camera')
        self._object = self._model.body(object_name)
        # Center the object at the origin.
        self._object.ipos = np.array([0,0,0])


        self._figsize = figsize
        self._renderer = mujoco.Renderer(self._model, 
                                        height=figsize[0], 
                                        width=figsize[1])

    @property
    def figsize(self)->Tuple[int,int]:
        return self._figsize

    @property
    def orientation(self)->npt.NDArray[np.float64]:
        # normalize the quaterniopn prior to returning it.
        quat = self._object.quat
        return quat / np.linalg.norm(quat)


    def render(self)->npt.NDArray[np.float64]:
        """
        Render the scene at the current position and orientation.

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            the rendered image.
        """
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera="camera")
        return self._renderer.render() 


    def set_orientation(self, quat:npt.NDArray[np.float64]) -> None:
        self._object.quat = quat


    def rotate_by(self, quat:npt.NDArray[np.float64]) -> None:
        """
        Rotate the object by the given quaternion.

        Parameters
        ----------
        quat : np.ndarray
            the quaternion to rotate by.
        """
        mujoco.mju_mulQuat(
            self._object.quat, quat, self._object.quat)
        

    def set_color(self, color:npt.NDArray[np.float64]) -> None:
        """
        Set the color of the object.

        Parameters
        ----------
        color : np.ndarray
            the rgba color to set.
        """
        self._model.geom(self._object.geomadr).rgba = color    


    def _create_xml(self, objects_dir:str, object_name:str):
        """
        Create the xml file for the mujoco simulation.
        """
        xml = f"""
        <mujoco>
            <visual>
                <quality numslices="1000" offsamples="1000"/>
            </visual>
            <asset>
                <texture type="skybox" builtin="flat" rgb1="1 1 1" width="32" height="512"/>
                <mesh name="{object_name}_mesh" file="{os.path.join(objects_dir,object_name)}.obj"/>
            </asset>

            <worldbody>
                <body name="{object_name}" pos="0 0 0">
                    <geom name="{object_name}" type="mesh" mesh="{object_name}_mesh" size=".2 .2 .2" rgba="1 0 0 1" quat="0.707 0.707 0 0"/>
                </body>
                <body name="camera" pos="0 0 0" quat="0 0 0 0">
                    <camera name="camera" mode="fixed" pos = "0 -0.32 0.0" quat="0.7 0.7 0 0"/>
                </body>
            </worldbody>
            <option timestep="0.01" gravity="0 0 0"/>
        </mujoco>
        """
        return xml
