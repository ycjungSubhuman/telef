/**
 * This file is part of YaPLY (Yet Another PLY library).
 *
 * Copyright (C) 2016 Alex Locher <alocher at ethz dot ch> (ETH Zuerich)
 * For more information see <https://github.com/alexlocher/yaply>
 *
 */
#ifndef INCLUDE_YAPLY_HPP_
#define INCLUDE_YAPLY_HPP_

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <exception>

namespace yaply {

    typedef uint32_t PLY_ELEMENT_UINT32;
    typedef float PLY_ELEMENT_FLOAT;

    template<typename T> inline const char* ply_type(); // not implemented

// specialized implementation for different types
#define DEFINE_PLY_TYPE( type , name ) \
template<>\
inline const char* ply_type<type>() {\
   return #name;\
}

// definitions according to http://paulbourke.net/dataformats/ply/
    DEFINE_PLY_TYPE(int8_t, char);
    DEFINE_PLY_TYPE(uint8_t, uchar);
    DEFINE_PLY_TYPE(int16_t, short);
    DEFINE_PLY_TYPE(uint16_t, ushort);
    DEFINE_PLY_TYPE(int32_t, int);
    DEFINE_PLY_TYPE(uint32_t, uchar);
    DEFINE_PLY_TYPE(float, float);
    DEFINE_PLY_TYPE(double, double);

    enum PLY_FORMAT {
        unknown, ascii, binary_le, binary_be
    };

    inline std::vector<std::string> split(std::string str, char delimiter) {
        std::vector<std::string> internal;
        std::stringstream ss(str); // Turn the string into a stream.
        std::string tok;

        while (getline(ss, tok, delimiter)) {
            internal.push_back(std::move(tok));
        }

        return internal;
    }

    class PLY_PROPERTY {
    public:
        virtual ~PLY_PROPERTY() {
        }

        std::string name;
        virtual bool load(std::istream& istr, size_t nr, bool binary) = 0;
        virtual void print_header(std::ostream& ostr) const = 0;
        virtual void print_data(std::ostream& ostr, const size_t nr, bool binary) const = 0;
    };

    template<typename SCALAR>
    class PLY_PROPERTY_SCALAR: public PLY_PROPERTY {
    public:
        PLY_PROPERTY_SCALAR(const std::string& name_, const size_t nrData_) {
            name = name_;
            data.resize(nrData_);
        }

        virtual ~PLY_PROPERTY_SCALAR() {
        }

        std::vector<SCALAR> data;

        inline SCALAR value(size_t nr) const {
            return data[nr];
        }

        virtual bool load(std::istream& istr, size_t nr, bool binary) {
            if (binary)
                istr.read((char*)(&data[nr]), sizeof(SCALAR));
            else
                istr >> data[nr];
            return istr.good();
        }

        virtual void print_header(std::ostream& ostr) const {
            ostr << "property " << ply_type<SCALAR>() << " " << name << std::endl;
        }

        virtual void print_data(std::ostream& ostr, const size_t nr, bool binary) const {
            if (binary)
                ostr.write((char*) &data[nr], sizeof(SCALAR));
            else
                ostr << data[nr] << " ";
        }

    };

// we have to specialize uchar and char for ascii (otherwise its not printed as a number)
    template<>
    inline void PLY_PROPERTY_SCALAR<unsigned char>::print_data(std::ostream& ostr, const size_t nr,
                                                               bool binary) const {
        if (binary)
            ostr.write((char*) &data[nr], 1);
        else
            ostr << (int) data[nr] << " ";
    }

    template<>
    inline void PLY_PROPERTY_SCALAR<char>::print_data(std::ostream& ostr, const size_t nr, bool binary) const {
        if (binary)
            ostr.write((char*) &data[nr], 1);
        else
            ostr << (int) data[nr] << " ";
    }

    template<>
    inline bool PLY_PROPERTY_SCALAR<uint8_t>::load(std::istream& istr, size_t nr, bool binary) {
        if (binary)
            istr.read((char*) &data[nr], 1);
        else {
            int tmp;
            istr >> tmp;
            data[nr] = (uint8_t) tmp;
        }
        return istr.good();
    }

    template<typename SCALAR, typename LIST_SCALAR>
    class PLY_PROPERTY_LIST: public PLY_PROPERTY {
    public:
        PLY_PROPERTY_LIST(const std::string& name_, const size_t nrData_) {
            name = name_;
            data.resize(nrData_);
        }

        virtual ~PLY_PROPERTY_LIST() {
        }

        std::vector<std::vector<SCALAR> > data;

        virtual bool load(std::istream& istr, size_t nr, bool binary) {
            LIST_SCALAR nrElements;
            if (binary) {
                istr.read((char*) (&nrElements), sizeof(LIST_SCALAR));
                for (LIST_SCALAR ii = 0; ii < nrElements; ii++)
                    istr.read((char*) (&data[nr][ii]), sizeof(SCALAR));
            } else {
                istr >> nrElements;
                data[nr].resize(nrElements);
                for (LIST_SCALAR ii = 0; ii < nrElements; ii++)
                    istr >> data[nr][ii];
            }
            return istr.good();
        }

        virtual void print_header(std::ostream& ostr) const {
            ostr << "property list " << ply_type<LIST_SCALAR>() << " " << ply_type<SCALAR>() << " "
                 << name << std::endl;
        }

        virtual void print_data(std::ostream& ostr, const size_t nr, bool binary) const {
            LIST_SCALAR size(data[nr].size());
            if (binary) {
                ostr.write((char*) &size, sizeof(LIST_SCALAR));
                for (SCALAR s : data[nr])
                    ostr.write((char*) &s, sizeof(SCALAR));
            } else {
                ostr << size << " ";
                for (SCALAR s : data[nr])
                    ostr << s << " ";
            }
        }

    };

    class PLY_ELEMENT {
    public:
        PLY_ELEMENT(const char* element_name, const size_t nrElements_ = 0) :
                name(element_name), nrElements(nrElements_) {
        }
        std::string name;
        size_t nrElements;
        std::vector<std::shared_ptr<PLY_PROPERTY> > properties;

        bool load(std::istream& istr, bool binary) {
            for (size_t ii = 0; ii < nrElements; ii++) {
                for (auto& p : properties) {
                    if (!p->load(istr, ii, binary))
                        return false;
                }
            }
            return true;
        }

        template<typename TYPE>
        void setScalars(const char* names, const TYPE* data);

        template<typename TYPE>
        void setList(const char* name, const std::vector<std::vector<TYPE>>& data);

        // gets a referenc to a property with the provided name / type (creates one if not yet existing!)
        template<class PROPERTY_TYPE>
        PROPERTY_TYPE& getPropertyCreate(const char* name);

        // gets a pointer to a property if it exists (otherwise nullptr)
        template<class PROPERTY_TYPE>
        PROPERTY_TYPE* getProperty(const char* name);

        // copies the elements of a scalar property if it exists
        template<typename SCALAR>
        bool getScalarProperty(const char* name, std::vector<SCALAR>& p);

        bool existsProperty(const char* name) const;
        bool existsProperties(const char* name) const;
    };

    template<typename TYPE>
    void PLY_ELEMENT::setScalars(const char* names, const TYPE* data) {
        std::vector<std::string> property_names = split(names, ',');
        const int nrProperties = property_names.size();
        for (int ii = 0; ii < nrProperties; ii++) {
            PLY_PROPERTY_SCALAR<TYPE>& property = getPropertyCreate<PLY_PROPERTY_SCALAR<TYPE>>(
                    property_names[ii].c_str());
            const TYPE* src_ptr = data + ii;
            for (int jj = 0; jj < nrElements; jj++, src_ptr += nrProperties) {
                property.data[jj] = *src_ptr;
            }
        }
    }

    template<typename TYPE>
    void PLY_ELEMENT::setList(const char* name, const std::vector<std::vector<TYPE>>& data) {
        PLY_PROPERTY_LIST<TYPE, uint32_t>& property = getPropertyCreate<
                                                      PLY_PROPERTY_LIST<TYPE, uint32_t>>(name);
        if (nrElements != data.size()) {
           throw std::runtime_error("nr Elements in List do not match");
        }
        for (size_t ii = 0; ii < nrElements; ii++){
            property.data[ii] = data[ii];
        }
    }

    template<class PROPERTY_TYPE>
    PROPERTY_TYPE& PLY_ELEMENT::getPropertyCreate(const char* name) {
        PROPERTY_TYPE* pointer = getProperty<PROPERTY_TYPE>(name);
        if (pointer != nullptr)
            return *pointer;

        // ok, not found, create one
        properties.emplace_back(std::shared_ptr<PROPERTY_TYPE>(new PROPERTY_TYPE(name, nrElements)));
        return *dynamic_cast<PROPERTY_TYPE*>(properties.back().get());
    }

    template<class PROPERTY_TYPE>
    PROPERTY_TYPE* PLY_ELEMENT::getProperty(const char* name) {
        for (auto& property : properties) {
            if (property->name.compare(name) == 0) {
                return dynamic_cast<PROPERTY_TYPE*>(property.get());
            }
        }
        return nullptr;
    }

    template<typename SCALAR>
    bool PLY_ELEMENT::getScalarProperty(const char* name, std::vector<SCALAR>& p){
        PLY_PROPERTY_SCALAR<SCALAR>* property = getProperty<PLY_PROPERTY_SCALAR<SCALAR>>(name);
        if (!property) return false;
        p = property->data;
        return true;
    }

    inline bool PLY_ELEMENT::existsProperty(const char* name) const {
        for (const auto& property : properties) {
            if (property->name.compare(name) == 0) {
                return true;
            }
        }
        return false;
    }

    inline bool PLY_ELEMENT::existsProperties(const char* name) const {
        std::vector<std::string> names = split(name, ',');
        for (const auto& n : names) {
            if (!existsProperty(n.c_str()))
                return false;
        }
        return true;
    }

    inline std::shared_ptr<PLY_PROPERTY> make_property(const std::string& line, const size_t nrElements) {
        std::shared_ptr<PLY_PROPERTY> p(nullptr);

        std::vector<std::string> tok = split(line, ' ');

        if (tok.size() == 5 && tok[1].compare("list") == 0) {
            // new list element
            if (tok[2].compare("uint") == 0 && tok[3].compare("uint") == 0) {
                p.reset(new PLY_PROPERTY_LIST<uint32_t, uint32_t>(tok[4], nrElements));
            } else if (tok[2].compare("int") == 0 && tok[3].compare("int") == 0) {
                p.reset(new PLY_PROPERTY_LIST<int32_t, int32_t>(tok[4], nrElements));
            } else if (tok[2].compare("uchar") == 0 && tok[3].compare("int") == 0) {
                p.reset(new PLY_PROPERTY_LIST<int32_t, int32_t>(tok[4], nrElements));
            } else {
                std::cerr << "cannot handle list element <" << line << ">" << std::endl;
                return p;
            }
        } else if (tok.size() == 3) {
            // new scalar element
            if (tok[1].compare("char") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<int8_t>(tok[2],nrElements));
            } else if (tok[1].compare("uchar") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<uint8_t>(tok[2],nrElements));
            } else if (tok[1].compare("short") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<int16_t>(tok[2],nrElements));
            } else if (tok[1].compare("ushort") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<uint16_t>(tok[2],nrElements));
            } else if (tok[1].compare("int") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<int32_t>(tok[2],nrElements));
            } else if (tok[1].compare("uint") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<uint32_t>(tok[2],nrElements));
            } else if (tok[1].compare("float") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<float>(tok[2],nrElements));
            } else if (tok[1].compare("double") == 0) {
                p.reset(new PLY_PROPERTY_SCALAR<double>(tok[2],nrElements));
            } else {
                std::cerr << "unknown scalar property <" << line << ">" << std::endl;
                return p;
            }
        } else {
            std::cerr << "Invalid ply file [property definition <" << line << ">]" << std::endl;
            return p;
        }
        return p;
    }

    class PlyFile {
    public:
        PlyFile() {
        }

        PlyFile(const char* path) {
            if (!load(path))
                std::cerr << "cannot load plyfile from >" << path << "<" << std::endl;
        }

        ~PlyFile() {
        }

        std::vector<PLY_ELEMENT> elements_;

        PLY_ELEMENT& operator[](const std::string& name);

        void save(const char* fname, bool binary) const;
        bool load(const std::string& fname);

//	template<typename TYPE>
//	void set(const char* name, const TYPE* data);
    };

    inline PLY_ELEMENT& PlyFile::operator[](const std::string& name) {
        for (PLY_ELEMENT& element : elements_) {
            if (name.compare(element.name) == 0)
                return element;
        }
        elements_.emplace_back(name.c_str());
        return elements_.back();
    }

    inline void PlyFile::save(const char* fname, bool binary) const {

        // test endianness
        int32_t n = 0;
        bool bigEndian = *(char *) &n == 1;

        // output the header
        std::ofstream pFile(fname, std::ofstream::out);
        pFile << "ply" << std::endl;
        pFile << std::setprecision (std::numeric_limits<double>::digits10 + 1);
        if (binary && bigEndian)
            pFile << "format binary_big_endian 1.0" << std::endl;
        else if (binary)
            pFile << "format binary_little_endian 1.0" << std::endl;
        else
            pFile << "format ascii 1.0" << std::endl;

        // print the elements
        for (const auto& elp : elements_) {
            pFile << "element " << elp.name << " " << elp.nrElements << std::endl;
            for (const auto& prop : elp.properties) {
                prop->print_header(pFile);
            }
        }
        pFile << "end_header" << std::endl;
        pFile.close();

        // output the points
        std::ofstream pData(fname,
                            (binary ? std::ofstream::binary | std::ofstream::app : std::ofstream::app));

        for (const auto& elp : elements_) {
            for (size_t ii = 0; ii < elp.nrElements; ii++) {
                for (const auto& prop : elp.properties) {
                    prop->print_data(pData, ii, binary);

                }
                if (!binary)
                    pData << std::endl;
            }
        }

        pData.flush();
        pData.close();
    }

    inline bool PlyFile::load(const std::string& file) {

        PLY_FORMAT ply_format = PLY_FORMAT::unknown;
        elements_.clear();

        // parse ply
        std::ifstream infile(file);
        std::string line;
        while (std::getline(infile, line)) {
            // skip comments
            if (line.compare(0, 7, "comment") == 0)
                continue;

            std::vector<std::string> tok = split(line, ' ');
            if (tok.size() < 1)
                continue;

            if (tok[0].compare("format") == 0) {
                if (tok.size() > 2) {
                    if (tok[1].compare("ascii") == 0)
                        ply_format = PLY_FORMAT::ascii;
                    else if (tok[1].compare("binary_big_endian") == 0)
                        ply_format = PLY_FORMAT::binary_be;
                    else if (tok[1].compare("binary_little_endian") == 0)
                        ply_format = PLY_FORMAT::binary_le;
                    continue;
                }

                std::cerr << "Invalid format declaration <" << line << ">" << std::endl;
                return false;
            }

            if (tok[0].compare("element") == 0) {
                if (tok.size() != 3) {
                    std::cerr << "Invalid ply file [element definition <" << line << ">]" << std::endl;
                    return false;
                }

                // create a new element
                elements_.emplace_back(tok[1].c_str(), std::atoi(tok[2].c_str()));
                continue;
            }

            if (tok[0].compare("property") == 0) {
                if (elements_.size() == 0) {
                    std::cerr << "missing element definition before <" << line << ">" << std::endl;
                    return false;
                }
                auto property = make_property(line,elements_.back().nrElements);
                if (property)
                    elements_.back().properties.emplace_back(std::move(property));
                else {
                    std::cerr << "failed to load property <" << line << ">" << std::endl;
                    return false;
                }
                continue;
            }

            // finished?
            if (tok[0].compare("end_header") == 0)
                break;
        }

        // alright, we should have the complete header now
        if (ply_format == PLY_FORMAT::unknown) {
            std::cerr << "Ply format not known";
            return false;
        }

        // now load the data into the prepared structure
        for (auto& el : elements_) {
            if (!el.load(infile, ply_format != PLY_FORMAT::ascii)) {
                return false;
            }
        }

        return true;
    }

} // end namespace

#endif /* INCLUDE_YAPLY_HPP_ */

