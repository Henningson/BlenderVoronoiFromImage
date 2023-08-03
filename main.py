import bpy
import bmesh
import numpy as np
import scipy.spatial as spatial
from random import random
from mathutils import Vector, Matrix
import colorsys
from math import sin, cos, pi
TAU = 2*pi
import colorsys
import os
import cv2



# This code is based on the code published by njanakiev (https://github.com/njanakiev)
# You can find the original here: https://github.com/njanakiev/blender-scripting/blob/master/scripts/voronoi_landscape.py


def remove_all(type=None):
    # Possible type:
    # "MESH", "CURVE", "SURFACE", "META", "FONT", "ARMATURE",
    # "LATTICE", "EMPTY", "CAMERA", "LIGHT"
    if type:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type=type)
        bpy.ops.object.delete()
    else:
        # Remove all elements in scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)


def create_material(base_color=(1, 1, 1, 1), metalic=0.0, roughness=0.5):
    mat = bpy.data.materials.new('Material')

    if len(base_color) == 3:
        base_color = list(base_color)
        base_color.append(1)

    mat.use_nodes = True
    node = mat.node_tree.nodes[0]
    node.inputs[0].default_value = base_color
    node.inputs[4].default_value = metalic
    node.inputs[7].default_value = roughness

    return mat 


def bmesh_to_object(bm, name='Object'):
    mesh = bpy.data.meshes.new(name + 'Mesh')
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    return obj



# Convert hsv values to gamma corrected rgb values
# Based on: http://stackoverflow.com/questions/17910543/convert-gamma-rgb-curves-to-rgb-curves-with-no-gamma/
def convert_hsv(hsv):
    color = [pow(val, 2.2) for val in colorsys.hsv_to_rgb(*hsv)]
    color.append(1)
    return color


def delaunay_landscape(img, points):
    vor = spatial.Delaunay(points)
    regions = vor.simplices
    verts = vor.points
    
    regions_filtered = []
    for region in regions:
        triangle_vertices = verts[region]
        triangle_vertices = np.array(triangle_vertices)
        triangle_vertices += 1.0
        triangle_vertices /= 2.0
        
        triangle_vertices = (triangle_vertices * np.expand_dims(np.array(img.shape), 0))
        triangle_centroid = (triangle_vertices.sum(axis=0) / 3.0).astype(np.int32)
        
        if (img[triangle_centroid[1], triangle_centroid[0]]) > 0.1:
            regions_filtered.append(region)
            
    regions = regions_filtered
    
    return verts, regions


def voronoi_landscape(img, points):
    # Create voronoi structure"
    vor = spatial.Voronoi(points)
    verts, regions = vor.vertices, vor.regions

    # Filter unused voronoi regions
    regions = [region for region in regions
                if not -1 in region and len(region) > 0]
    regions = [region for region in regions
                if np.all([np.linalg.norm(verts[i]) < 1.0 for i in region])]

    regions_filtered = []
    for region in regions:
        triangle_vertices = verts[region]
        triangle_vertices = np.array(triangle_vertices)
        triangle_vertices += 1.0
        triangle_vertices /= 2.0
        
        triangle_vertices = (triangle_vertices * np.expand_dims(np.array(img.shape), 0))
        triangle_centroid = (triangle_vertices.sum(axis=0) / triangle_vertices.shape[0]).astype(np.int32)
        
        if (img[triangle_centroid[1], triangle_centroid[0]]) > 0.1:
            regions_filtered.append(region)
            
    regions = regions_filtered
    
    return verts, regions


def get_colors_from_range(color_range=[[0.5, 0.7], [0.7, 0.8], [0.8, 0.9]], num_colors=20):
    colors = np.random.random((num_colors, 3))
    for i, r in zip(range(num_colors), color_range):
        colors[:, i] = (r[1] - r[0])*colors[:, i] + r[0]
    
    return colors    


def point_cloud_from_image(image, n=50, keep_index=10):
    contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    
    contour_points = np.concatenate(contours).squeeze()
    contours_filtered = contour_points[::keep_index, :]
    
    norm_contours = contours_filtered - np.array(image.shape)[None, :]//2
    image = image.astype(np.float32) / 255.0
    image /= image.sum()
    
    candidates = np.arange(0, image.size)
    chosen_points = np.random.choice(candidates, n, p=image.flatten(), replace=False)
    
    
    points = np.zeros(image.shape)
    points[contours_filtered[:, 1], contours_filtered[:, 0]] = 1.0
        
    points = points.flatten()
    points[chosen_points] = 1.0
    points = points.reshape(image.shape)
    points_imagespace = np.stack(points.nonzero(), axis=-1)[:, [1,0]].astype(np.float32)
    
    points_normalized = (points_imagespace / np.expand_dims(image.shape, 0)) * 2.0 - 1.0
    
    return points_normalized


def mesh_from_regions(verts, regions, w=20, h=2):    
    # Create faces from voronoi regions
    bm = bmesh.new()
    vDict, faces = {}, []
    for region in regions:
        for idx in region:
            if not idx in vDict:
                x, y, z = verts[idx, 0]*w, verts[idx, 1]*w, 0
                vert = bm.verts.new((x, y, z))
                vDict[idx] = vert

        face = bm.faces.new(tuple(vDict[i] for i in region))
        faces.append(face)

    bmesh.ops.recalc_face_normals(bm, faces=faces)

    # Extrude faces randomly
    top_faces = []
    for face in faces:
        r = bmesh.ops.extrude_discrete_faces(bm, faces=[face])
        f = r['faces'][0]
        top_faces.append(f)
        bmesh.ops.translate(bm, vec=Vector((0, 0, random()*h)), verts=f.verts)
        center = f.calc_center_bounds()
        bmesh.ops.scale(bm, vec=Vector((0.8, 0.8, 0.8)), verts=f.verts, space=Matrix.Translation(-center))


    # Assign material index to each bar
    for face in top_faces:
        idx = np.random.randint(len(colors))
        face.material_index = idx
        for edge in face.edges:
            for f in edge.link_faces:
                f.material_index = idx

    # Create obj and mesh from bmesh object
    me = bpy.data.meshes.new("VoronoiMesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("Voronoi", me)
    bpy.context.scene.collection.objects.link(obj)

    # Create and assign materials to object
    for color in colors:
        mat = create_material(convert_hsv(color))
        obj.data.materials.append(mat)


if __name__ == '__main__':
    # Remove all elements
    remove_all()

    colors = get_colors_from_range()
    
    path = "PATH_TO_YOUR_IMAGE"
    image = cv2.imread(path, 0)

    voronoi_points = point_cloud_from_image(image, n=400, keep_index=10)
    verts, regions = voronoi_landscape(image, voronoi_points)
    mesh_from_regions(verts, regions)
    
    delaunay_points = point_cloud_from_image(image, n=50, keep_index=50)
    verts, regions = delaunay_landscape(image, delaunay_points)
    mesh_from_regions(verts, regions)