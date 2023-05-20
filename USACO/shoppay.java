import java.util.*;
import java.io.*;
import java.lang.*;

public class shoppay
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("shoppay.in"));
      int n = in.nextInt();
      Queue<Integer> line = new LinkedList<>();
      Queue<Integer>[] registers = new Queue[n];
      
      for(int i = 0; i < n; i++)
         registers[i] = new LinkedList<>();
      
      while(in.hasNext())
      {
         if(in.next().equals("C"))
            line.add(in.nextInt());
         else
         {
            int reg = in.nextInt() - 1;
            registers[reg].add(line.remove());
         }
      }
      
      in.close();
     
      for(int i = 0; i < n; i++)
      {
         int m = registers[i].size();
         System.out.print(m);
         for(int j = 0; j < m; j++)
            System.out.print(" " + registers[i].remove());
         
         System.out.println();
      }
   }
}