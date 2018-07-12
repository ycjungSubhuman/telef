#include <boost/make_shared.hpp>
#include <pcl/PolygonMesh.h>
#include <pcl/TextureMesh.h>
#include "io/align/PCAVisualizerPrepPipe.h"
#include "mesh/colormapping.h"

#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>

#include "io/bmp.h"

using namespace telef::align;

namespace telef::io::align {
    boost::shared_ptr<PCAVisualizerSuite>
    PCAVisualizerPrepPipe::_processData(boost::shared_ptr<PCANonRigidFittingResult> in)
    {
        auto result = boost::make_shared<PCAVisualizerSuite>();

        auto pcaFitMesh = in->pca_model->genMesh(in->fitCoeff);
        pcaFitMesh.applyTransform(in->transformation);

        projectColor(in->image, pcaFitMesh, in->fx, in->fy);


        pcl::PCLImage::Ptr pclImage = boost::make_shared<pcl::PCLImage>();

        /*
         * Convert Image to PCLImage
         */
        // Set Header
        pclImage->header.frame_id = in->image->getFrameID();
//        pclImage->header.seq
        pclImage->header.stamp = in->image->getTimestamp();
        // Set Metadata
        pclImage->height = in->image->getHeight();
        pclImage->width = in->image->getWidth();
        pclImage->encoding = in->image->getEncoding();
//        pclImage->is_bigendian
        pclImage->step = in->image->getStep();

        // set data
        uint8_t *dataRef = new uint8_t[in->image->getDataSize()];
        in->image->fillRaw(dataRef);

        pclImage->data.assign(dataRef, dataRef + in->image->getDataSize());

        delete [] dataRef;


        pcl::PointCloud<pcl::PointXYZ> _pcaPtCld;
        telef::util::convert(_pcaPtCld, pcaFitMesh.position);

        pcl::TextureMesh::Ptr mesh = boost::make_shared<pcl::TextureMesh>();
        pcl::toPCLPointCloud2(_pcaPtCld, mesh->cloud);

        // Create the texturemesh object that will contain our UV-mapped mesh
//        pcl::PolygonMesh::Ptr triangles = boost::make_shared<pcl::PolygonMesh>();
//        triangles->cloud = pcaPtCld;


        std::vector< pcl::Vertices> polygons;

        // push faces into the texture mesh object
        for(size_t i =0; i < pcaFitMesh.triangles.size(); ++i)
        {
            pcl::Vertices face;
            face.vertices.assign(pcaFitMesh.triangles[i].begin(), pcaFitMesh.triangles[i].end());
            polygons.push_back(face);
        }
        mesh->tex_polygons.push_back(polygons);

        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> >  tex_coordinates;
        for(size_t i = 0; i < pcaFitMesh.uv.rows()/2; i+=2) {
            Eigen::Vector2f tex_cood(pcaFitMesh.uv[2*i], pcaFitMesh.uv[2*i+1]);
            tex_coordinates.push_back(tex_cood);
        }
        mesh->tex_coordinates.push_back(tex_coordinates);

        // Create materials for each texture
        /*
         * tex_materials:
         * Ka 1.000000 1.000000 1.000000
         * Kd 1.000000 1.000000 1.000000
         * Ks 0.000000 0.000000 0.000000
         * Tr 1.000000
         * illum 1
         * Ns 0.000000
         * map_Kd jake2.jpg
         */
        pcl::TexMaterial mesh_material;
        mesh_material.tex_Ka.r = 1.0f;
        mesh_material.tex_Ka.g = 1.0f;
        mesh_material.tex_Ka.b = 1.0f;

        mesh_material.tex_Kd.r = 1.0f;
        mesh_material.tex_Kd.g = 1.0f;
        mesh_material.tex_Kd.b = 1.0f;

        mesh_material.tex_Ks.r = 0.0f;
        mesh_material.tex_Ks.g = 0.0f;
        mesh_material.tex_Ks.b = 0.0f;

        mesh_material.tex_d = 1.0f; // Transparency
        mesh_material.tex_Ns = 0.0f;
        mesh_material.tex_illum = 1;

        std::stringstream tex_name;
        tex_name << "material_" << 0;
        tex_name >> mesh_material.tex_name;

        fs::path p{outputPath.c_str()};
        auto stripped = p.parent_path()/p.stem();
        mesh_material.tex_file = stripped.string()+".jpg";

        mesh->tex_materials.push_back(mesh_material);

        // Assign texture image
        mesh->texture = pclImage;


        PCL_INFO ("\tInput mesh contains %d faces and %d vertices\n", mesh->tex_polygons[0].size(), pcaFitMesh.triangles.size());
        PCL_INFO ("...Done.\n");

        //pcl::io::savePLYFile("test_mesh_ascii.ply", *mesh);

        //telef::io::saveBMPFile(stripped.string() + ".jpg", *in->image);
//        pcl::io::saveOBJFile(stripped.string()+".obj", *mesh);

        result->model = mesh;

        return result;
    }

//    int
//    PCAVisualizerPrepPipe::createMesh(pcl::TextureMesh &mesh, const ColorMesh &colorMesh, pcl::PCLImage &tex_image)
//    {
//        Eigen::Vector4f origin;
//        Eigen::Quaternionf orientation;
//        int file_version;
//        const int offset = 1;
//
//        pcl::console::TicToc tt;
//        tt.tic ();
//
//        int data_type;
//        unsigned int data_idx;
//        if (readHeader (file_name, mesh.cloud, origin, orientation, file_version, data_type, data_idx, offset))
//        {
//            PCL_ERROR ("[pcl::OBJReader::read] Problem reading header!\n");
//            return (-1);
//        }
//
//        std::ifstream fs;
//        fs.open (file_name.c_str (), std::ios::binary);
//        if (!fs.is_open () || fs.fail ())
//        {
//            PCL_ERROR ("[pcl::OBJReader::readHeader] Could not open file '%s'! Error : %s\n",
//                       file_name.c_str (), strerror(errno));
//            fs.close ();
//            return (-1);
//        }
//
//        // Seek at the given offset
//        fs.seekg (data_idx, std::ios::beg);
//
//        // Get normal_x and rgba fields indices
//        int normal_x_field = -1;
//        // std::size_t rgba_field = 0;
//        for (std::size_t i = 0; i < mesh.cloud.fields.size (); ++i)
//            if (mesh.cloud.fields[i].name == "normal_x")
//            {
//                normal_x_field = i;
//                break;
//            }
//
//        std::size_t v_idx = 0;
//        std::size_t vn_idx = 0;
//        std::size_t vt_idx = 0;
//        std::size_t f_idx = 0;
//        std::string line;
//        std::vector<std::string> st;
//        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > coordinates;
//        try
//        {
//            while (!fs.eof ())
//            {
//                getline (fs, line);
//                // Ignore empty lines
//                if (line == "")
//                    continue;
//
//                // Tokenize the line
//                std::stringstream sstream (line);
//                sstream.imbue (std::locale::classic ());
//                line = sstream.str ();
//                boost::trim (line);
//                boost::split (st, line, boost::is_any_of ("\t\r "), boost::token_compress_on);
//
//                // Ignore comments
//                if (st[0] == "#")
//                    continue;
//                // Vertex
//                if (st[0] == "v")
//                {
//                    try
//                    {
//                        for (int i = 1, f = 0; i < 4; ++i, ++f)
//                        {
//                            float value = boost::lexical_cast<float> (st[i]);
//                            memcpy (&mesh.cloud.data[v_idx * mesh.cloud.point_step + mesh.cloud.fields[f].offset],
//                                    &value,
//                                    sizeof (float));
//                        }
//                        ++v_idx;
//                    }
//                    catch (const boost::bad_lexical_cast &e)
//                    {
//                        PCL_ERROR ("Unable to convert %s to vertex coordinates!", line.c_str ());
//                        return (-1);
//                    }
//                    continue;
//                }
//                // Vertex normal
//                if (st[0] == "vn")
//                {
//                    try
//                    {
//                        for (int i = 1, f = normal_x_field; i < 4; ++i, ++f)
//                        {
//                            float value = boost::lexical_cast<float> (st[i]);
//                            memcpy (&mesh.cloud.data[vn_idx * mesh.cloud.point_step + mesh.cloud.fields[f].offset],
//                                    &value,
//                                    sizeof (float));
//                        }
//                        ++vn_idx;
//                    }
//                    catch (const boost::bad_lexical_cast &e)
//                    {
//                        PCL_ERROR ("Unable to convert line %s to vertex normal!", line.c_str ());
//                        return (-1);
//                    }
//                    continue;
//                }
//                // Texture coordinates
//                if (st[0] == "vt")
//                {
//                    try
//                    {
//                        Eigen::Vector3f c (0, 0, 0);
//                        for (std::size_t i = 1; i < st.size (); ++i)
//                            c[i-1] = boost::lexical_cast<float> (st[i]);
//                        if (c[2] == 0)
//                            coordinates.push_back (Eigen::Vector2f (c[0], c[1]));
//                        else
//                            coordinates.push_back (Eigen::Vector2f (c[0]/c[2], c[1]/c[2]));
//                        ++vt_idx;
//                    }
//                    catch (const boost::bad_lexical_cast &e)
//                    {
//                        PCL_ERROR ("Unable to convert line %s to texture coordinates!", line.c_str ());
//                        return (-1);
//                    }
//                    continue;
//                }
//                // Material
//                if (st[0] == "usemtl")
//                {
//                    mesh.tex_polygons.push_back (std::vector<pcl::Vertices> ());
//                    mesh.tex_materials.push_back (pcl::TexMaterial ());
//                    for (std::size_t i = 0; i < companions_.size (); ++i)
//                    {
//                        std::vector<pcl::TexMaterial>::const_iterator mat_it = companions_[i].getMaterial (st[1]);
//                        if (mat_it != companions_[i].materials_.end ())
//                        {
//                            mesh.tex_materials.back () = *mat_it;
//                            break;
//                        }
//                    }
//                    // We didn't find the appropriate material so we create it here with name only.
//                    if (mesh.tex_materials.back ().tex_name == "")
//                        mesh.tex_materials.back ().tex_name = st[1];
//                    mesh.tex_coordinates.push_back (coordinates);
//                    coordinates.clear ();
//                    continue;
//                }
//                // Face
//                if (st[0] == "f")
//                {
//                    //We only care for vertices indices
//                    pcl::Vertices face_v; face_v.vertices.resize (st.size () - 1);
//                    for (std::size_t i = 1; i < st.size (); ++i)
//                    {
//                        int v;
//                        sscanf (st[i].c_str (), "%d", &v);
//                        v = (v < 0) ? v_idx + v : v - 1;
//                        face_v.vertices[i-1] = v;
//                    }
//                    mesh.tex_polygons.back ().push_back (face_v);
//                    ++f_idx;
//                    continue;
//                }
//            }
//        }
//        catch (const char *exception)
//        {
//            PCL_ERROR ("[pcl::OBJReader::read] %s\n", exception);
//            fs.close ();
//            return (-1);
//        }
//
//        double total_time = tt.toc ();
//        PCL_DEBUG ("[pcl::OBJReader::read] Loaded %s as a TextureMesh in %g ms with %g points, %g texture materials, %g polygons.\n",
//                   file_name.c_str (), total_time,
//                   v_idx -1, mesh.tex_materials.size (), f_idx -1);
//        fs.close ();
//        return (0);
//    }
}
