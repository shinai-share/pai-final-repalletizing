"""Opt-in presentation study. No collision, mass, joint or friction edits.

Generated textures are illustrative, not measured product surface data.
Visual additions can affect camera observations: never reuse perception verdicts.
"""
from pathlib import Path
import numpy as np


def textures(directory):
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(40)
    n = 2048
    for name, color, rough in [('kraft', (163,125,78), .88), ('concrete',(117,122,119),.76),
                                ('paint',(210,166,30),.38), ('steel',(143,153,160),.28)]:
        grain = rng.normal(0, 2.8 if name == 'kraft' else 1.5, (n,n))
        coarse = np.asarray(Image.fromarray(rng.integers(90,160,(64,64),dtype=np.uint8)).resize((n,n)).filter(ImageFilter.GaussianBlur(12)),float)-125
        rgb = np.clip(np.array(color)[None,None,:]+grain[:,:,None]+coarse[:,:,None]*.18,0,255).astype('uint8')
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        if name == 'kraft':
            # Folded edges, printed handling graphics and fibre-scale paper variation.
            d.rectangle((5,5,n-6,n-6),outline=(108,77,46),width=7)
            d.rectangle((16,16,n-17,n-17),outline=(186,147,94),width=3)
            font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',72)
            small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',38)
            d.text((130,220),'PACKAGED GOODS',fill=(58,51,41),font=font)
            d.text((134,325),'SIMULATION SAMPLE',fill=(65,58,45),font=small)   # each case's mass differs: none printed
            for x in (180,330):
                d.line((x,690,x,490),fill=(56,50,40),width=16)
                d.polygon([(x,460),(x-40,520),(x+40,520)],fill=(56,50,40))
            d.line((140,720,370,720),fill=(56,50,40),width=10)
            d.rectangle((1250,1250,1890,1750),fill=(216,210,191))
            for x in range(1280,1850,9):
                if rng.random()>.25:d.rectangle((x,1320,x+int(rng.integers(2,7)),1580),fill=(41,43,38))
            d.text((1280,1650),'LOT 024 / DEMO',fill=(41,43,38),font=small)
        if name == 'concrete':
            d.line((0,0,n,0),fill=(68,73,71),width=9)
            d.line((0,0,0,n),fill=(68,73,71),width=9)
            for _ in range(900):
                x,y=rng.integers(0,n,2); r=int(rng.integers(1,4))
                d.ellipse((x,y,x+r,y+r),fill=(93,99,95))
        im.save(directory/f'{name}.png')
        Image.fromarray(np.clip(rough*255+grain*1.4,0,255).astype('uint8')).save(directory/f'{name}_rough.png')
    return directory


def apply(stage, directory, cell=None):
    from pxr import UsdGeom, UsdShade, Sdf, Gf, UsdLux
    from . import usd_build as ub, cell as cm
    directory = Path(directory).resolve()
    def physics_signature():
        return {str(p.GetPath())+'/'+a.GetName():str(a.Get()) for p in stage.Traverse()
                for a in p.GetAttributes() if a.GetName().startswith(('physics:','physx','drive:','limit:'))}
    physics_before=physics_signature()
    mats={}
    for name, metal in [('kraft',0.),('concrete',0.),('paint',.25),('steel',.92)]:
        path=f'/World/visual_detail/materials/{name}'
        mat=ub.material(stage,path,(.5,.5,.5),roughness=.6,metallic=metal)
        shader=UsdShade.Shader(stage.GetPrimAtPath(path+'/Shader'))
        uv=UsdShade.Shader.Define(stage,path+'/UV')
        uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        for channel, filename, output in [('diffuseColor',name+'.png','rgb'),('roughness',name+'_rough.png','r')]:
            tex=UsdShade.Shader.Define(stage,path+'/'+channel)
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(directory/filename)))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if channel=='diffuseColor' else 'raw')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
            tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
            shader.GetInput(channel).ConnectToSource(tex.ConnectableAPI(),output)
        mats[name]=mat
    # Visual meshes duplicate boxes' exact extents and transforms. Original colliders stay intact.
    faces=[[(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)],
           [(-1,1,-1),(-1,-1,-1),(-1,-1,1),(-1,1,1)],
           [(-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)],
           [(1,-1,-1),(1,-1,1),(-1,-1,1),(-1,-1,-1)],
           [(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)],
           [(-1,1,-1),(1,1,-1),(1,-1,-1),(-1,-1,-1)]]
    count=0
    for prim in list(stage.Traverse()):
        if prim.GetTypeName()!='Cube':continue
        p=str(prim.GetPath()); name=None
        if '/cases/' in p and p.endswith('/geom'):name='kraft'
        elif p.endswith('/floor/geom'):name='concrete'
        elif '/gantry/' in p:
            bound=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            material_name=bound.GetPrim().GetName() if bound else ''
            if material_name=='frame':name='paint'
            elif material_name=='fork_bar':name='steel'
        elif '/scenery/rack_' in p and 'stock' in p:name='kraft'
        if name is None:continue
        size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        mesh=UsdGeom.Mesh.Define(stage,p+'_visual')
        mesh.CreatePointsAttr([Gf.Vec3f(*(np.array(v)*size/2)) for face in faces for v in face])
        mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr(list(range(24)))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        xf=UsdGeom.Xformable(mesh)
        xf.AddTransformOp().Set(UsdGeom.Xformable(prim).GetLocalTransformation())
        tile=8. if name=='concrete' else 1.
        st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying')
        # Each side's vertical UV direction follows world-local Z; top remains a flat print.
        uv=[]
        for k,face in enumerate(faces):
            for x,y,z in face:
                u,v=((y,z) if k<2 else (x,z) if k<4 else (x,y))
                uv.append(Gf.Vec2f((u+1)*.5*tile,(v+1)*.5*tile))
        st.Set(uv)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[name])
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set('invisible')
        count+=1
        if name in ('paint','steel') and '/gantry/' in p:
            transform=UsdGeom.Xformable(prim).GetLocalTransformation()
            # Small illustrative fastener heads mounted to the existing member, no collision.
            for j,(x,z) in enumerate([(-.34,-.34),(.34,-.34),(-.34,.34),(.34,.34)]):
                position=transform.Transform(Gf.Vec3d(x*size,-.505*size,z*size))
                bolt=UsdGeom.Cylinder.Define(stage,p+f'_fastener_{j}')
                bolt.CreateRadiusAttr(.0045)
                bolt.CreateHeightAttr(.004)
                bolt.CreateAxisAttr('Y')
                ub.set_pose(bolt.GetPrim(),tuple(position))
                UsdShade.MaterialBindingAPI.Apply(bolt.GetPrim()).Bind(mats['steel'])
    # Fine decking grooves on the existing pallet top (visual only).
    dark=ub.material(stage,'/World/visual_detail/materials/groove',(.045,.05,.052),roughness=.7)
    for pallet in ('pallet_a','pallet_b'):
        path=f'/World/cell/{pallet}'
        geom=stage.GetPrimAtPath(path+'/geom')
        scale=geom.GetAttribute('xformOp:scale').Get()
        if scale:
            w,d,h=map(float,scale)
            for i in range(1,12):
                ub.box(stage,path+f'/deck_groove_{i}',(.002,d*.94,.0002),
                       pos=(-w/2+w*i/12,0,h/2+.00015),mat=dark,collide=False)
    # Replace saturated blue top print with a translucent-looking packing-tape finish.
    tape=UsdShade.Shader(stage.GetPrimAtPath('/World/cell/mat/case_band/Shader'))
    tape.GetInput('diffuseColor').Set(Gf.Vec3f(.43,.30,.15))
    tape.GetInput('roughness').Set(.22)
    # Soft overhead illumination instead of a dominant low sun.
    UsdLux.DistantLight(stage.GetPrimAtPath('/World/sun')).GetIntensityAttr().Set(160.)
    UsdLux.DistantLight(stage.GetPrimAtPath('/World/sun')).GetAngleAttr().Set(15.)
    UsdLux.DomeLight(stage.GetPrimAtPath('/World/dome')).GetIntensityAttr().Set(500.)
    fill=UsdLux.RectLight.Define(stage,'/World/visual_detail/front_softbox')
    fill.CreateWidthAttr(4.);fill.CreateHeightAttr(3.);fill.CreateIntensityAttr(2200.)
    ub.set_pose(fill.GetPrim(),(1.,-3.,4.),quat=cm._look_quat((1.,-3.,4.),(0.,0.,.5)))
    # Three-quarter view makes the depth of the two pallets and the mechanism legible.
    eye=(2.6,-5.2,3.2); target=(0.,0.,.85)
    cam=UsdGeom.Camera(stage.GetPrimAtPath(cm.CAMERAS['presentation']))
    ub.set_pose(cam.GetPrim(),eye,quat=cm._look_quat(eye,target))
    cam.GetFocalLengthAttr().Set(20.)
    dressed=0
    if cell is not None:
        for i in dressing(cell):
            parent='/World/visual_detail/dress' if i['parent']=='world' else f"{cm.GANTRY}/{i['parent']}"
            path=f"{parent}/dress_{i['group']}_{i['name']}"
            mat=ub.material(stage,f"/World/visual_detail/materials/dress_{i['group']}_{i['name']}",i['color'],roughness=.5,metallic=.3)
            if i['shape']=='cylinder':
                c=UsdGeom.Cylinder.Define(stage,path)
                c.CreateRadiusAttr(i['size'][0]/2);c.CreateHeightAttr(i['size'][2]);c.CreateAxisAttr('Z')
                ub.set_pose(c.GetPrim(),i['pos'])
                UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
            else:
                ub.box(stage,path,i['size'],pos=i['pos'],mat=mat,collide=False)
            dressed+=1
    physics_unchanged=physics_before==physics_signature()
    if not physics_unchanged:raise RuntimeError('visual detail changed physics attributes')
    return {'visual_meshes':count,'dressing_items':dressed,'textures':'illustrative procedural, not measured',
            'physics_attributes_unchanged':physics_unchanged,'perception_revalidation_required':True}


CLEAR = 0.45              # m between the gantry's legs and the cabinet and posts


def dressing(cell):
    """What makes the cell look like an installed machine, for the picture only: a stack light, a control cabinet,
    servo motors on the axes they drive, and cable ducts. No fence and no light-curtain posts (2026-11-01: not wanted
    in the picture). Positions are world coordinates, or the moving body's own frame for
    'base', 'bridge', 'trolley' and 'ram'. Nothing here collides or has mass."""
    g = cell.gantry
    leg_x = (g.x_range[1] - g.x_range[0] + 0.6) / 2
    leg_y = (g.y_range[1] - g.y_range[0] + 0.6) / 2
    fx, fy_back = leg_x + CLEAR, leg_y + CLEAR
    top = g.beam_height - g.origin_height                   # beam top in the gantry bodies' frame
    dark, grey = (0.10, 0.10, 0.11), (0.62, 0.64, 0.66)
    items = []

    def add(group, name, size, pos, color, parent='world', shape='box'):
        items.append({'group': group, 'name': name, 'size': tuple(size), 'pos': tuple(pos), 'color': color,
                      'parent': parent, 'shape': shape})

    lamp = (leg_x, -leg_y, g.beam_height + 0.02)              # stack light on the front-right leg
    for k, color in enumerate(((0.25, 0.25, 0.27), (0.75, 0.10, 0.08), (0.90, 0.62, 0.05), (0.15, 0.85, 0.30))):
        add('stack_light', f'tier_{k}', (0.09, 0.09, 0.07), (lamp[0], lamp[1], lamp[2] + 0.04 + 0.075 * k), color,
            shape='cylinder')
    cab = (-fx, fy_back - 0.1)                               # control cabinet beyond the back-left leg
    add('cabinet', 'body', (0.6, 0.4, 1.8), (cab[0], cab[1], 0.9), grey)
    add('cabinet', 'hmi', (0.32, 0.012, 0.22), (cab[0], cab[1] - 0.206, 1.35), dark)
    add('cabinet', 'stop', (0.05, 0.05, 0.03), (cab[0] + 0.2, cab[1] - 0.215, 1.15), (0.8, 0.08, 0.06), shape='cylinder')
    for sy in (-1, 1):                                       # X drives at both ends of the bridge
        add('motor', f'x_drive_{"np"[sy > 0]}', (0.16, 0.16, 0.22), (0.0, sy * (leg_y - 0.05), top + 0.02), dark,
            parent='bridge')
    add('motor', 'y_drive', (0.14, 0.14, 0.20), (0.25, 0.0, 0.82), dark, parent='trolley')
    add('motor', 'z_drive', (0.16, 0.16, 0.18), (0.0, 0.0, 2.73), dark, parent='ram')
    add('duct', 'x_duct', (2 * leg_x, 0.10, 0.10), (0.0, leg_y + 0.12, top), dark, parent='base')
    add('duct', 'y_duct', (0.08, 2 * leg_y, 0.08), (0.14, 0.0, top - 0.18), dark, parent='bridge')
    return items
